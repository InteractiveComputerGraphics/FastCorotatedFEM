#include "Common.h"
#include "utilities/MiniGL.h"
#include "utilities/Timing.h"
#include "TetModel.h"
#include "GL/glut.h"

INIT_TIMING

void timeStep();
void buildModel();
void render();
void reset();
void pauseSimulation();

const unsigned int TIMESTEP = 10; // in msec
bool moving = false;
TetModel tetModel;

// main 
int main(int argc, char **argv)
{
	REPORT_MEMORY_LEAKS

	// OpenGL
	MiniGL::init(argc, argv, 1024, 768, 0, 0, "FastCorotDemo");
	MiniGL::initLights();
	MiniGL::initTexture();
	MiniGL::setViewport(40.0, 0.1f, 500.0, Vector3r(0.0, 0.2, 1.0), Vector3r(0.0, 0.0, 0.0));
	
	// OpenGL
	MiniGL::setClientIdleFunc(50, timeStep);
	MiniGL::setKeyFunc(0, 'r', reset);
	MiniGL::setKeyFunc(1, ' ', pauseSimulation);
	MiniGL::setClientSceneFunc(render);

	buildModel();

	glutMainLoop();

	Utilities::Timing::printAverageTimes();
	Utilities::Timing::printTimeSums();

	return 0;
}

void pauseSimulation()
{
	moving = !moving;
}

void timeStep()
{
    if (moving == true)
	{
		START_TIMING("Simulation step");
		tetModel.simulator.step(tetModel.x, tetModel.v, tetModel.ind_tets, TIMESTEP/1000.0);
		STOP_TIMING_AVG;
    }
}

void buildModel()
{
	Real density = 1000.0;
	Real E = 1.0e6;
	Real nu = 0.33;
	Real mu = E / (2.0 * (1.0 + nu));
	Real lambda = (E*nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
	std::string nodeFileName = std::string(DATA_PATH) + "armadillo_4k.node";
	std::string eleFileName = std::string(DATA_PATH) + "armadillo_4k.ele";
	std::vector<anchor> anchors;
	anchors.push_back(anchor(811, 0.02));
	Real angle = 180.0;

	tetModel = TetModel(nodeFileName, eleFileName, Vector3r(0,0,0), 0.1, angle, Vector3r(0,1,0), density, mu, lambda, TIMESTEP/1000.0, anchors);
}

void render() 
{
	MiniGL::coordinateSystem();

	float color[4] = { 0.0f, 0.2f, 0.6f, 1.0f };
	float speccolor[4] = { 0.25, 0.25, 0.25, 0.25 };
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, speccolor);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0);

	//draw 4 triangles per tet
	if (tetModel.ind_tets.size() > 0)
	{
		glBegin(GL_TRIANGLES);
		for (int i = 0; i < tetModel.nTets; i++)
		{
			const Vector3r& x0 = tetModel.x[tetModel.ind_tets[i][0]];
			const Vector3r& x1 = tetModel.x[tetModel.ind_tets[i][1]];
			const Vector3r& x2 = tetModel.x[tetModel.ind_tets[i][2]];
			const Vector3r& x3 = tetModel.x[tetModel.ind_tets[i][3]];
	
			Vector3r n = -(x1 - x0).cross(x2 - x0);
			glNormal3dv(n.data());
			glVertex3dv(x0.data());
			glVertex3dv(x1.data());
			glVertex3dv(x2.data());
	
			n = (x1 - x0).cross(x3 - x0);
			glNormal3dv(n.data());
			glVertex3dv(x0.data());
			glVertex3dv(x1.data());
			glVertex3dv(x3.data());
	
			n = -(x2 - x0).cross(x3 - x0);
			glNormal3dv(n.data());
			glVertex3dv(x0.data());
			glVertex3dv(x2.data());
			glVertex3dv(x3.data());
	
			n = -(x1 - x2).cross(x3 - x2);
			glNormal3dv(n.data());
			glVertex3dv(x2.data());
			glVertex3dv(x1.data());
			glVertex3dv(x3.data());
		}
		glEnd();
	}

	glDisable(GL_LIGHTING);
	float strCol[4] = { 0.0, 0.0, 0.0, 1.0 };
	std::string timeStr = "Time: ";
	timeStr = timeStr + std::to_string(tetModel.simulator.time);
	MiniGL::drawBitmapText(-0.95f, 0.9f, timeStr.c_str(), (int) timeStr.size(), strCol);
	glEnable(GL_LIGHTING);
}

void reset()
{
	moving = false;
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::reset();
	buildModel();
}