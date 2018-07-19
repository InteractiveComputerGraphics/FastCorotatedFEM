#include "TetModel.h"

using namespace std;

// ----------------------------------------------------------------------------------------------
TetModel::TetModel(string filename_node, string filename_ele,
	Vector3r t, Real scale, Real angle, Vector3r axis,
	Real density, Real mu, Real lambda, Real dt, vector<anchor>& anchors)
{
	vector<unsigned int> tets;
	TetGenLoader::loadTetgenModel(filename_node, filename_ele, x, tets);
	Quaternionr initialRotation(AngleAxisr(angle * M_PI / 180.0, axis));
	Matrix3r R = initialRotation.matrix();

	for (int i = 0; i < static_cast<int>(x.size()); i++)
		x[i] = R * x[i] * scale + t;
	
	ind_tets.resize(tets.size() / 4);
	for (int i = 0; i< tets.size() / 4; i++)
	{
		ind_tets[i].resize(4);
		ind_tets[i][0] = (int)tets[4 * i + 0];
		ind_tets[i][1] = (int)tets[4 * i + 1];
		ind_tets[i][2] = (int)tets[4 * i + 2];
		ind_tets[i][3] = (int)tets[4 * i + 3];
	}

	nVerts = (int)x.size();
	nTets = (int)ind_tets.size();
	x_0 = x;
	v = vector<Vector3r>(nVerts, Vector3r(0, 0, 0));

	nFixedVerts = 0;
	fixVertices(anchors);
	
	simulator.initialize(x, ind_tets, Quaternionr::Identity(), nFixedVerts, density, mu, lambda, dt);
}

// ----------------------------------------------------------------------------------------------
//sorts the vertices such that the fixed vertices are grouped at the beginning of the 
//positions array
void TetModel::fixVertices(vector<anchor> &anchors)
{
	nFixedVerts = 0;
	for (anchor& a : anchors)
	{
		const Vector3r fixCenter = x[a.centralVertex];
		for (int i = 0; i < nVerts; i++)
		{
			if ((x[i] - fixCenter).norm() < a.radius)
			{
				int id = i;
				if (i > nFixedVerts) {
					swapParticles(i, nFixedVerts);
					id = nFixedVerts;
					for (anchor& a2 : anchors)
						if (a2.centralVertex == nFixedVerts)
							a2.centralVertex = i;
				}
				nFixedVerts++;
				a.fixedVertices.push_back(id);
			}
		}
	}
}

// ----------------------------------------------------------------------------------------------
//swaps the (rest)positions of the particles at positions i0 and i1
//and swaps their indices in the index array
void TetModel::swapParticles(int i0, int i1)
{
	swap(x[i0], x[i1]);
	swap(x_0[i0], x_0[i1]);

	for (int i = 0; i<ind_tets.size(); i++)
		for (int j = 0; j<4; j++)
		{
			if (ind_tets[i][j] == i0)
			{
				ind_tets[i][j] = i1;
				continue;
			}
			if (ind_tets[i][j] == i1) ind_tets[i][j] = i0;
		}
}


