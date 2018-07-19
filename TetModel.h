#ifndef TETMODEL_H
#define TETMODEL_H

#include "Common.h"
#include "utilities/TetGenLoader.h"
#include "FastCorotFEM.h"
#include <vector>

struct anchor;

class TetModel{
public:
	TetModel(){}
	TetModel(std::string filename_node, std::string filename_ele,
			Vector3r t, Real scale, Real angle, Vector3r axis,
			Real density, Real mu, Real lambda, Real dt, std::vector<anchor>& anchors);
	
	std::vector<Vector3r> x;	//vertices of the tetMesh
	std::vector<Vector3r> x_0;	//vertices of the tetMesh in rest pose
	std::vector<Vector3r> v;	//velocities
	std::vector<std::vector<int>> ind_tets;	//indices of the tets

	int nVerts;
	int nTets;
	int nFixedVerts;

	FastCorotFEM simulator;

	void fixVertices(std::vector<anchor> &anchors);
	void swapParticles(int i0, int i1);
};

struct anchor	//for fixing some vertices of the tetmesh
{
	anchor() { radius = 0.0; }
	anchor(int centralVertex, Real radius)
	{
		this->centralVertex = centralVertex;
		this->radius = radius;
	}

	int centralVertex;
	Real radius;
	std::vector<int> fixedVertices;
};

#endif //TETMODEL_H
