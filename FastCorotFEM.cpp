#include "FastCorotFEM.h"

using namespace std;
using namespace Eigen;

// ----------------------------------------------------------------------------------------------
//Performs one time step of the simulation with Alg. 2 from the paper
void FastCorotFEM::step(
		vector<Vector3r> &x,
		vector<Vector3r> &v,
		const vector<vector<int>> &ind,
		Real dt)
{
	//explicit Euler to compute \tilde{x}
	for (int i = nFixedVertices; i < nVerts; i++)
	{
		v[i].y() -= dt * 9.81;	//gravity
		x_old[i] = x[i];
		x[i] += dt * v[i];
	}
	
	solveOptimizationProblem(x, ind);

	//solve volume constraints 
	for(size_t i=0; i<kappa_phases.size(); i++)	//reset Lagrange multipliers
		for (size_t j=0; j< kappa_phases[i].size(); j++)
			kappa_phases[i][j] = Scalarf8(0.0f);

	for(int it = 0; it < 2; it++)	//solve constraints
		solveVolumeConstraints(x, ind);

	//velocity update
	for (int i = nFixedVertices; i < nVerts; i++)
		v[i] = (x[i] - x_old[i]) / dt;

	time += dt;
}

// ----------------------------------------------------------------------------------------------
//Initializes the simulation with Alg. 1 from the paper
bool FastCorotFEM::initialize(
		const vector<Vector3r> &p,
		const vector<vector<int>> &ind,
		const Quaternionr& initialRotation,
		int nFixedVertices,
		Real density,
		Real mu,
		Real lambda,
		Real dt)
{
	time = 0.0;
	nVerts = static_cast<int>(p.size());
	nTets = static_cast<int>(ind.size());
	if (nTets % 8 == 0) vecSize = nTets / 8;
	else vecSize = nTets / 8 + 1;
	this->nFixedVertices = nFixedVertices;
			
	vector<Triplet<Real>> triplets_D;
	triplets_D.reserve(9 * nTets * 4);
	vector<Triplet<Real>> triplets_K;
	triplets_K.reserve(9 * nTets);
	vector<Triplet<Real>> triplets_M;
	triplets_M.reserve(4 * nTets);
	vector<Real> Kreal(nTets);
	vector<Real[4][3]> Dt(nTets);
	vector<Matrix3r> Dm_inv(nTets);
	vector<Real> rest_volume(nTets);
	vector<Real> invMass(nVerts);

	//Algorithm 1, lines 1-12
	for (int t = 0; t < nTets; t++)
	{
		//indices of the 4 vertices of tet t
		int it[4] = { ind[t][0], ind[t][1], ind[t][2], ind[t][3] };

		//compute rest pose shape matrix and volume
		Matrix3r Dm;
		Dm.col(0) = p[it[1]] - p[it[0]];
		Dm.col(1) = p[it[2]] - p[it[0]];
		Dm.col(2) = p[it[3]] - p[it[0]];
		
		rest_volume[t] = 1.0 / 6.0 * Dm.determinant();
		if (rest_volume[t] < 0.0) return false;
		Dm_inv[t] = Dm.inverse();
		
		//set triplets for the matrix K. Directly multiply the factor 2*dt*dt into K
		Kreal[t] = 2.0 * dt * dt * mu * rest_volume[t];

		for (int j = 0; j < 9; j++)
			triplets_K.push_back(Triplet<Real>(9 * t + j, 9 * t + j, Kreal[t]));
		
		//initialize the lumped mass matrix
		for (int j = 0; j < 4; j++)		//forall verts of tet i
		{
			invMass[it[j]] += 0.25 * density * rest_volume[t];
			triplets_M.push_back(Triplet<Real>(it[j], it[j], invMass[it[j]]));
		}
		
		//compute matrix D_t from Eq. (9) (actually Dt[t] is D_t^T)
		for (int k = 0; k < 3; k++)
			Dt[t][0][k] = -Dm_inv[t](0, k) - Dm_inv[t](1, k) - Dm_inv[t](2, k);

		for (int j = 1; j < 4; j++)
			for (int k = 0; k < 3; k++)
				Dt[t][j][k] = Dm_inv[t](j - 1, k);

		//initialize the matrix D
		for (int i = 0; i<4; i++)
			for (int j = 0; j < 3; j++)
				triplets_D.push_back(Triplet<Real>(9 * t + 3 * j, it[i], Dt[t][i][j]));
	}

	//set matrices
	SparseMatrix<Real> K(9 * nTets, 9 * nTets);	// actually 2 * dt* dt * K
	SparseMatrix<Real> D(9 * nTets, nVerts);
	SparseMatrix<Real> M(nVerts, nVerts);
	K.setFromTriplets(triplets_K.begin(), triplets_K.end());
	D.setFromTriplets(triplets_D.begin(), triplets_D.end());
	M.setFromTriplets(triplets_M.begin(), triplets_M.end());
	
	//compute system matrix and Cholesky factorization (Algorithm 1, line 13)
	//remove the upper-left 3*nFixedVertices x 3*nFixedVertices block
	SparseMatrix<Real> M_plus_DT_K_D = (M + D.transpose() * K * D).block(
		nFixedVertices, nFixedVertices, nVerts - nFixedVertices, nVerts - nFixedVertices);
	
	SimplicialLLT<SparseMatrix<Real>, Lower, AMDOrdering<int>> LLT;
	LLT.compute(M_plus_DT_K_D);
	perm = LLT.permutationP();
	permInv = LLT.permutationPinv();
	matL = SparseMatrix<float, ColMajor>(LLT.matrixL().cast<float>());
	matLT = SparseMatrix<float, ColMajor>(LLT.matrixU().cast<float>());
	
	//move data to vector registers
	Kvec.resize(vecSize);
	convertToAVX(Kreal, Kvec);
	DT.resize(vecSize);
	convertToAVX(Dt, DT);
	//prepare solver variables
	x_old.resize(nVerts);
	quats.resize(vecSize);
	for (int i = 0; i < vecSize; i++) 
		quats[i] = Quaternion8f((float)initialRotation.x(), (float)initialRotation.y(), 
								(float)initialRotation.z(), (float)initialRotation.w());
	RHS.resize(nVerts - nFixedVertices);
	RHS_perm.resize(nVerts - nFixedVertices);

	//initialize volume constraints
	for (size_t i = 0; i < invMass.size(); i++)
		if (i >= nFixedVertices && invMass[i] != 0.0)
			invMass[i] = 1.0 / invMass[i];
		else
			invMass[i] = 0.0;

	initializeVolumeConstraints(ind, rest_volume, invMass, lambda, dt);

	return true;
}

// ----------------------------------------------------------------------------------------------
//initializes the volume constraints. For parallel Gauss-Seidel they are grouped with graph coloring
//the inverse masses, alpha values and rest volumes are moved to vector registers
void FastCorotFEM::initializeVolumeConstraints(
		const vector<vector<int>> &ind,
		vector<Real> &rest_volume,
		vector<Real> &invMass,
		Real lambda,
		Real dt)
{
	constraintGraphColoring(ind, nVerts, volume_constraint_phases);
	
	inv_mass_phases.resize(volume_constraint_phases.size());
	rest_volume_phases.resize(volume_constraint_phases.size());
	kappa_phases.resize(volume_constraint_phases.size());
	alpha_phases.resize(volume_constraint_phases.size());

	for (int phase = 0; phase < volume_constraint_phases.size(); phase++)	//forall constraint phases
	{
		inv_mass_phases[phase].resize(0);
		rest_volume_phases[phase].resize(0);
		kappa_phases[phase].resize(0);
		alpha_phases[phase].resize(0);
		for (int c = 0; c<volume_constraint_phases[phase].size(); c += 8)	//forall constraints in phase
		{
			int c8[8];
			for (int k = 0; k<8; k++)
				if (c + k < volume_constraint_phases[phase].size())
					c8[k] = volume_constraint_phases[phase][c + k];

			float w0[8], w1[8], w2[8], w3[8], vol[8], alpha[8];
			for (int k = 0; k < 8; k++)
				if (c + k < volume_constraint_phases[phase].size())
				{
					w0[k] = (float)invMass[ind[c8[k]][0]];
					w1[k] = (float)invMass[ind[c8[k]][1]];
					w2[k] = (float)invMass[ind[c8[k]][2]];
					w3[k] = (float)invMass[ind[c8[k]][3]];

					vol[k] = (float)rest_volume[c8[k]];
					alpha[k] = 1.0f / (float)(lambda * rest_volume[c8[k]] * dt * dt);	
				}
				else
				{
					vol[k] = 1.0f;
					alpha[k] = 0.0f;
					w0[k] = (float)invMass[ind[c8[k]][0]];
					w1[k] = (float)invMass[ind[c8[k]][1]];
					w2[k] = (float)invMass[ind[c8[k]][2]];
					w3[k] = (float)invMass[ind[c8[k]][3]];
				}
			
			int pos = (int)inv_mass_phases[phase].size();
			inv_mass_phases[phase].push_back(vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>(4));

			inv_mass_phases[phase][pos][0].load(w0);
			inv_mass_phases[phase][pos][1].load(w1);
			inv_mass_phases[phase][pos][2].load(w2);
			inv_mass_phases[phase][pos][3].load(w3);

			Scalarf8 restVol, alpha8;
			restVol.load(vol);
			alpha8.load(alpha);
			rest_volume_phases[phase].push_back(restVol);
			kappa_phases[phase].push_back(Scalarf8(0.0f));
			alpha_phases[phase].push_back(alpha8);
		}
	}
}

// ----------------------------------------------------------------------------------------------
// this method is taken from the PBD library: https://github.com/InteractiveComputerGraphics/PositionBasedDynamics
void FastCorotFEM::constraintGraphColoring(
		const vector<vector<int>>& particleIndices, int n,
		vector<vector<int>>& coloring) 
{
	//particleIndices [constraint][particleIndex]
	vector<vector<bool>> particleColors;   //numColors x numParticles, true if particle in color
	particleColors.resize(0);
	coloring.resize(0);

	for (unsigned int i = 0; i < particleIndices.size(); i++)   //forall constraints
	{
		bool newColor = true;
		for (unsigned int j = 0; j < coloring.size(); j++)  //forall colors
		{
			bool addToThisColor = true;

			for (unsigned int k = 0; k < particleIndices[i].size(); k++) { //forall particles innvolved in the constraint
				if (particleColors[j][particleIndices[i][k]] == true) {
					addToThisColor = false;
					break;
				}
			}
			if (addToThisColor) {
				coloring[j].push_back(i);

				for (unsigned int k = 0; k < particleIndices[i].size(); k++) //forall particles innvolved in the constraint
					particleColors[j][particleIndices[i][k]] = true;

				newColor = false;
				break;
			}
		}
		if (newColor) {
			particleColors.push_back(vector<bool>(n, false));
			coloring.resize(coloring.size() + 1);
			coloring[coloring.size() - 1].push_back(i);
			for (unsigned int k = 0; k < particleIndices[i].size(); k++) //forall particles innvolved in the constraint
				particleColors[coloring.size() - 1][particleIndices[i][k]] = true;
		}
	}
}

// ----------------------------------------------------------------------------------------------
//
void FastCorotFEM::solveOptimizationProblem(
		vector<Vector3r> &p,
		const vector<vector<int>> &ind)
{
	//compute RHS of Equation (12)
	for (size_t i = 0; i < RHS.size(); i++)
		RHS[i] = Scalarf8(0.0f);

	for (int i = 0; i < vecSize; i++)
	{
		Vector3f8 F1, F2, F3;	//columns of the deformation gradient
		computeDeformationGradient(p, ind, i, F1, F2, F3);

		Quaternion8f& q = quats[i];
		APD_Newton_AVX(F1, F2, F3, q);

		//transform quaternion to rotation matrix
		Vector3f8 R1, R2, R3;	//columns of the rotation matrix
		quats[i].toRotationMatrix(R1, R2, R3);
		
		// R <- R - F
		R1 -= F1;
		R2 -= F2;
		R3 -= F3;

		//multiply with 2 * dt * dt * DT * K from left
		Vector3f8 dx[4];
		dx[0] = (R1 * DT[i][0][0] + R2 * DT[i][0][1] + R3 * DT[i][0][2]) * Kvec[i];
		dx[1] = (R1 * DT[i][1][0] + R2 * DT[i][1][1] + R3 * DT[i][1][2]) * Kvec[i];
		dx[2] = (R1 * DT[i][2][0] + R2 * DT[i][2][1] + R3 * DT[i][2][2]) * Kvec[i];
		dx[3] = (R1 * DT[i][3][0] + R2 * DT[i][3][1] + R3 * DT[i][3][2]) * Kvec[i];
	
		//write results to the corresponding positions in the RHS vector
		for(int k = 0; k < 4; k++)
		{
			float x[8], y[8], z[8];
			dx[k].x().store(x);
			dx[k].y().store(y);
			dx[k].z().store(z);

			for (int j = 0; j < 8; j++)
			{
				if(8 * i + j >= nTets) break;
				int pi = ind[8 * i + j][k];
				if (pi < nFixedVertices) continue;
				RHS[pi - nFixedVertices] += Scalarf8(x[j], y[j], z[j], x[j], y[j], z[j], x[j], y[j]);	//only first 3 comps are used, maybe use 128 bit registers
			}
		}
	}

	//solve the linear system
	//permutation of the RHS because of Eigen's fill-in reduction
	for (size_t i = 0; i < RHS.size(); i++)
		RHS_perm[perm.indices()[i]] = RHS[i];

	//foreward substitution
	for (int k = 0; k<matL.outerSize(); ++k)
		for (SparseMatrix<float, ColMajor>::InnerIterator it(matL, k); it; ++it)
			if (it.row() == it.col())
				RHS_perm[it.row()] = RHS_perm[it.row()] / Scalarf8(it.value());
			else
				RHS_perm[it.row()] -= Scalarf8(it.value()) * RHS_perm[it.col()];
		
	//backward substitution
	for (int k = matLT.outerSize() - 1; k >= 0 ; --k)
		for (SparseMatrix<float, ColMajor>::ReverseInnerIterator it(matLT, k); it; --it)
			if (it.row() == it.col())
				RHS_perm[it.row()] = RHS_perm[it.row()] / Scalarf8(it.value());
			else
				RHS_perm[it.row()] -= Scalarf8(it.value()) * RHS_perm[it.col()];
	
	//invert permutation
	for (size_t i = 0; i < RHS.size(); i++)
		RHS[permInv.indices()[i]] = RHS_perm[i];
	
	for (size_t i = 0; i<RHS.size(); i++)	// add result (delta_x) to the positions
	{
		float x[8];
		RHS[i].store(x);
		p[i + nFixedVertices].x() += x[0];
		p[i + nFixedVertices].y() += x[1];
		p[i + nFixedVertices].z() += x[2];
	}
}

// ----------------------------------------------------------------------------------------------
//computes the deformation gradient of 8 tets
inline void FastCorotFEM::computeDeformationGradient(
		const vector<Vector3r> &p, 
		const vector<vector<int>> &ind, int i,
		Vector3f8 & F1, Vector3f8 & F2, Vector3f8 & F3)
{
	Vector3f8 vertices[4];	//vertices of 8 tets
	int regularPart = (nTets / 8) * 8;
	int i8 = 8 * i;
	
	if (i8 < regularPart)
	{
		for (int j = 0; j < 4; j++)
		{
			const Vector3r& p0_0 = p[ind[i8 + 0][j]];
			const Vector3r& p0_1 = p[ind[i8 + 1][j]];
			const Vector3r& p0_2 = p[ind[i8 + 2][j]];
			const Vector3r& p0_3 = p[ind[i8 + 3][j]];
			const Vector3r& p0_4 = p[ind[i8 + 4][j]];
			const Vector3r& p0_5 = p[ind[i8 + 5][j]];
			const Vector3r& p0_6 = p[ind[i8 + 6][j]];
			const Vector3r& p0_7 = p[ind[i8 + 7][j]];

			vertices[j].x() = Scalarf8(p0_0[0], p0_1[0], p0_2[0], p0_3[0], p0_4[0], p0_5[0], p0_6[0], p0_7[0]);
			vertices[j].y() = Scalarf8(p0_0[1], p0_1[1], p0_2[1], p0_3[1], p0_4[1], p0_5[1], p0_6[1], p0_7[1]);
			vertices[j].z() = Scalarf8(p0_0[2], p0_1[2], p0_2[2], p0_3[2], p0_4[2], p0_5[2], p0_6[2], p0_7[2]);
		}
	}
	else    //add padding with vertices of last tet. (they are never read out)
	{
		for (int j = 0; j < 4; j++)
		{
			Vector3f p0[8];
			for (int k = regularPart; k < regularPart + 8; k++)
				if (k < nTets) p0[k - regularPart] = p[ind[k][j]].cast<float>();
				else p0[k - regularPart] = p[ind[nTets - 1][j]].cast<float>();

				vertices[j].x() = Scalarf8(p0[0][0], p0[1][0], p0[2][0], p0[3][0], p0[4][0], p0[5][0], p0[6][0], p0[7][0]);
				vertices[j].y() = Scalarf8(p0[0][1], p0[1][1], p0[2][1], p0[3][1], p0[4][1], p0[5][1], p0[6][1], p0[7][1]);
				vertices[j].z() = Scalarf8(p0[0][2], p0[1][2], p0[2][2], p0[3][2], p0[4][2], p0[5][2], p0[6][2], p0[7][2]);
		}
	}

	// compute F as D_t*x (see Equation (9))
	F1 = vertices[0] * DT[i][0][0] + vertices[1] * DT[i][1][0] + vertices[2] * DT[i][2][0] + vertices[3] * DT[i][3][0];
	F2 = vertices[0] * DT[i][0][1] + vertices[1] * DT[i][1][1] + vertices[2] * DT[i][2][1] + vertices[3] * DT[i][3][1];
	F3 = vertices[0] * DT[i][0][2] + vertices[1] * DT[i][1][2] + vertices[2] * DT[i][2][2] + vertices[3] * DT[i][3][2];
}


// ----------------------------------------------------------------------------------------------
//computes the APD of 8 deformation gradients. (Alg. 3 from the paper)
inline void FastCorotFEM::APD_Newton_AVX(const Vector3f8& F1, const Vector3f8& F2, const Vector3f8& F3, Quaternion8f& q)
{
	//one iteration is sufficient for plausible results
	for (int it = 0; it<1; it++)
	{
		//transform quaternion to rotation matrix
		Matrix3f8 R;
		q.toRotationMatrix(R);

		//columns of B = RT * F
		Vector3f8 B0 = R.transpose() * F1;
		Vector3f8 B1 = R.transpose() * F2;
		Vector3f8 B2 = R.transpose() * F3;

		Vector3f8 gradient(B2[1] - B1[2], B0[2] - B2[0], B1[0] - B0[1]);

		//compute Hessian, use the fact that it is symmetric
		Scalarf8 h00 = B1[1] + B2[2];
		Scalarf8 h11 = B0[0] + B2[2];
		Scalarf8 h22 = B0[0] + B1[1];
		Scalarf8 h01 = Scalarf8(-0.5) * (B1[0] + B0[1]);
		Scalarf8 h02 = Scalarf8(-0.5) * (B2[0] + B0[2]);
		Scalarf8 h12 = Scalarf8(-0.5) * (B2[1] + B1[2]);

		Scalarf8 detH = Scalarf8(-1.0) * h02 * h02 * h11 + Scalarf8(2.0) * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;

		Vector3f8 omega;
		//compute symmetric inverse
		const Scalarf8 factor = Scalarf8(-0.25) / detH;
		omega[0] = (h11 * h22 - h12 * h12) * gradient[0]
			+ (h02 * h12 - h01 * h22) * gradient[1]
			+ (h01 * h12 - h02 * h11) * gradient[2];
		omega[0] *= factor;

		omega[1] = (h02 * h12 - h01 * h22) * gradient[0]
			+ (h00 * h22 - h02 * h02) * gradient[1]
			+ (h01 * h02 - h00 * h12) * gradient[2];
		omega[1] *= factor;

		omega[2] = (h01 * h12 - h02 * h11) * gradient[0]
			+ (h01 * h02 - h00 * h12) * gradient[1]
			+ (h00 * h11 - h01 * h01) * gradient[2];
		omega[2] *= factor;

		omega = Vector3f8::blend(abs(detH) < 1.0e-9f, gradient * Scalarf8(-1.0), omega);	//if det(H) = 0 use gradient descent, never happened in our tests, could also be removed 

		//instead of clamping just use gradient descent. also works fine and does not require the norm
		Scalarf8 useGD = blend(omega * gradient > Scalarf8(0.0), Scalarf8(1.0), Scalarf8(-1.0));
		omega = Vector3f8::blend(useGD > Scalarf8(0.0), gradient * Scalarf8(-0.125), omega);

		Scalarf8 l_omega2 = omega.lengthSquared();
		const Scalarf8 w = (1.0 - l_omega2) / (1.0 + l_omega2);
		const Vector3f8 vec = omega * (2.0 / (1.0 + l_omega2));
		q = q * Quaternion8f(vec.x(), vec.y(), vec.z(), w);		//no normalization needed because the Cayley map returs a unit quaternion
	}
}

// ----------------------------------------------------------------------------------------------
//
void FastCorotFEM::solveVolumeConstraints(
		vector<Vector3r> &x,
		const vector<vector<int>> &ind)
{
	for (int phase = 0; phase < volume_constraint_phases.size(); phase++)	//forall constraint phases
	{
		for (int constraint = 0; constraint < inv_mass_phases[phase].size(); constraint++)	//forall constraints in this phase
		{
			//move the positions of 8 tetrahedrons to vector registers
			Vector3f8 p[4];
			int c8[8];	//indices of 8 tets
			for (int k = 0; k < 8; k++)
				if (8 * constraint + k < volume_constraint_phases[phase].size())
					c8[k] = volume_constraint_phases[phase][8 * constraint + k];
				else
					c8[k] = 0;

			for (int j = 0; j < 4; j++)
			{
				const Vector3r& p0_0 = x[ind[c8[0]][j]];
				const Vector3r& p0_1 = x[ind[c8[1]][j]];
				const Vector3r& p0_2 = x[ind[c8[2]][j]];
				const Vector3r& p0_3 = x[ind[c8[3]][j]];
				const Vector3r& p0_4 = x[ind[c8[4]][j]];
				const Vector3r& p0_5 = x[ind[c8[5]][j]];
				const Vector3r& p0_6 = x[ind[c8[6]][j]];
				const Vector3r& p0_7 = x[ind[c8[7]][j]];

				p[j].x() = Scalarf8(p0_0[0], p0_1[0], p0_2[0], p0_3[0], p0_4[0], p0_5[0], p0_6[0], p0_7[0]);
				p[j].y() = Scalarf8(p0_0[1], p0_1[1], p0_2[1], p0_3[1], p0_4[1], p0_5[1], p0_6[1], p0_7[1]);
				p[j].z() = Scalarf8(p0_0[2], p0_1[2], p0_2[2], p0_3[2], p0_4[2], p0_5[2], p0_6[2], p0_7[2]);
			}

			//solve the constraints 
			const float eps = 1e-6f;

			//compute the volume using Eq. (14)
			Vector3f8 d1 = p[1] - p[0];
			Vector3f8 d2 = p[2] - p[0];
			Vector3f8 d3 = p[3] - p[0];
			Scalarf8 volume = (d1 % d2) * d3 * (1.0f / 6.0f);

			//compute the gradients (see: supplemental document)
			Vector3f8 grad1 = d2 % d3;
			Vector3f8 grad2 = d3 % d1;
			Vector3f8 grad3 = d1 % d2;
			Vector3f8 grad0 = -grad1 - grad2 - grad3;

			const Scalarf8& restVol = rest_volume_phases[phase][constraint];
			const Scalarf8& alpha = alpha_phases[phase][constraint];
			Scalarf8& kappa = kappa_phases[phase][constraint];

			//compute the Lagrange multiplier update using Eq. (15)
			Scalarf8 delta_kappa =
				inv_mass_phases[phase][constraint][0] * grad0.lengthSquared() +
				inv_mass_phases[phase][constraint][1] * grad1.lengthSquared() +
				inv_mass_phases[phase][constraint][2] * grad2.lengthSquared() +
				inv_mass_phases[phase][constraint][3] * grad3.lengthSquared() +
				alpha;
			
			delta_kappa = (restVol - volume - alpha * kappa) / blend(abs(delta_kappa) < eps, 1.0f, delta_kappa);
			kappa = kappa + delta_kappa;

			//compute the position updates using Eq. (16)
			p[0] = p[0] + grad0 * delta_kappa * inv_mass_phases[phase][constraint][0];
			p[1] = p[1] + grad1 * delta_kappa * inv_mass_phases[phase][constraint][1];
			p[2] = p[2] + grad2 * delta_kappa * inv_mass_phases[phase][constraint][2];
			p[3] = p[3] + grad3 * delta_kappa * inv_mass_phases[phase][constraint][3];

			//write the positions from the vector registers back to the positions array
			for (int j = 0; j < 4; j++)
			{
				float px[8], py[8], pz[8];
				p[j].x().store(px);
				p[j].y().store(py);
				p[j].z().store(pz);

				for (int k = 0; k < 8; k++)
					if (8 * constraint + k < volume_constraint_phases[phase].size())
						x[ind[c8[k]][j]] = Vector3r(px[k], py[k], pz[k]);
			}
		}
	}
}

// ----------------------------------------------------------------------------------------------
//
void FastCorotFEM::convertToAVX(const vector<Real>& v, vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>& vAVX)
{
	int regularPart = (nTets / 8) * 8;
	for (int i = 0; i < regularPart; i += 8)
		vAVX[i / 8] = Scalarf8(v[i + 0], v[i + 1], v[i + 2], v[i + 3], v[i + 4], v[i + 5], v[i + 6], v[i + 7]);
	
	if (regularPart != nTets)	//add padding with last value of v. (they are never read out)
		for (int j = 0; j < 3; j++)
		{
			Real vtmp[8];
			for (int i = regularPart; i < regularPart + 8; i++)
				if (i < nTets) vtmp[i - regularPart] = v[i];
				else vtmp[i - regularPart] = v[nTets - 1];

				vAVX[regularPart / 8] = Scalarf8(vtmp[0], vtmp[1], vtmp[2], vtmp[3], vtmp[4], vtmp[5], vtmp[6], vtmp[7]);
		}
}

// ----------------------------------------------------------------------------------------------
//
void FastCorotFEM::convertToAVX(
		const vector<Real[4][3]>& v, 
		vector<vector<vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>>>& vAVX)
{
	int regularPart = (nTets / 8) * 8;
	for (int i = 0; i < regularPart; i += 8)
	{
		vAVX[i / 8].resize(4);
		for (int j = 0; j < 4; j++)
		{
			vAVX[i / 8][j].resize(3);
			for (int k = 0; k < 3; k++)
				vAVX[i / 8][j][k] = Scalarf8(v[i + 0][j][k], v[i + 1][j][k], v[i + 2][j][k], v[i + 3][j][k], v[i + 4][j][k], v[i + 5][j][k], v[i + 6][j][k], v[i + 7][j][k]);
		}
	}

	if (regularPart != nTets) {	//add padding with last value of v. (they are never read out)
		vAVX[regularPart / 8].resize(4);
		for (int j = 0; j < 4; j++)
		{
			vAVX[regularPart / 8][j].resize(3);
			for (int k = 0; k < 3; k++)
			{
				Real vtmp[8];
				for (int i = regularPart; i < regularPart + 8; i++)
					if (i < nTets) vtmp[i - regularPart] = v[i][j][k];
					else vtmp[i - regularPart] = v[nTets - 1][j][k];

				vAVX[regularPart/ 8][j][k] = Scalarf8(vtmp[0],	vtmp[1], vtmp[2], vtmp[3], vtmp[4], vtmp[5], vtmp[6], vtmp[7]);
			}
		}
	}
}