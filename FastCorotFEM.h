#ifndef FASTCOROTFEM_H
#define FASTCOROTFEM_H

#include <vector>
#include "Eigen/Sparse"
#include "Common.h"
#include "AVX_math.h"

class FastCorotFEM
{
public:

	int nVerts;
	int nTets;
	int vecSize;
	int nFixedVertices;
	Real time;
		
	//Cholesky factorization of the system matrix
	Eigen::SparseMatrix<float, Eigen::ColMajor> matL;
	Eigen::SparseMatrix<float, Eigen::ColMajor> matLT;
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permInv;
	//temporal varialbes of the solver
	std::vector<Vector3r> x_old;
	std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>> RHS;
	std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>> RHS_perm;
	std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>> Kvec;
	std::vector<std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>>> DT;
	std::vector<Quaternion8f, AlignmentAllocator<Quaternion8f, 32> > quats;
	//variables for the volume constraints
	std::vector<std::vector<int>> volume_constraint_phases;
	std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>> rest_volume_phases;
	std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>> alpha_phases;
	std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>> kappa_phases;
	std::vector<std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>>> inv_mass_phases;
 
	FastCorotFEM(): nVerts(0), nTets(0), vecSize(0), nFixedVertices(0)
	{
	}

	void step(
			std::vector<Vector3r> &x,
			std::vector<Vector3r> &v,
			const std::vector<std::vector<int>> &ind,
			Real dt);

	bool initialize(
			const std::vector<Vector3r> &p,
			const std::vector<std::vector<int>> &ind,
			const Quaternionr& initialRotation,
			int nFixedVertices,
			Real denstiy,
			Real mu,
			Real lambda,
			Real dt);

	void initializeVolumeConstraints(
			const std::vector<std::vector<int>> &ind,
			std::vector<Real> &rest_volume,
			std::vector<Real> &invMass,
			Real lambda,
			Real dt);

	void constraintGraphColoring(
			const std::vector<std::vector<int>>& particleIndices, 
			int n,
			std::vector<std::vector<int>>& coloring);
	
	void solveOptimizationProblem(
			std::vector<Vector3r> &p,
			const std::vector<std::vector<int>> &ind);
	
	inline void computeDeformationGradient(
			const std::vector<Vector3r> &p,
			const std::vector<std::vector<int>> &ind, int i,
			Vector3f8 & F1, Vector3f8 & F2, Vector3f8 & F3);

	inline void APD_Newton_AVX(
			const Vector3f8& F1, 
			const Vector3f8& F2, 
			const Vector3f8& F3, 
			Quaternion8f& q);

	void solveVolumeConstraints(
			std::vector<Vector3r> &x,
			const std::vector<std::vector<int>> &ind);

	void convertToAVX(
			const std::vector<Real>& v, 
			std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>& vAVX);

	void convertToAVX(
			const std::vector<Real[4][3]>& v, 
			std::vector<std::vector<std::vector<Scalarf8, AlignmentAllocator<Scalarf8, 32>>>>& vAVX);
};
#endif