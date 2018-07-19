#ifndef AVX_MATH_H
#define AVX_MATH_H

#include <ctime>
#include <iomanip>
#include <vector>
#include <memory>
#include <immintrin.h>

#ifdef __linux__
#include <malloc.h>
#endif

#include "Common.h"

// ----------------------------------------------------------------------------------------------
//vector of 8 float values to represent 8 scalars
class Scalarf8
{
public:
	__m256 v; 

	Scalarf8() {}

	Scalarf8(float f) {	v = _mm256_set1_ps(f); }

	Scalarf8(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
		v = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
	}

	Scalarf8(Real f0, Real f1, Real f2, Real f3, Real f4, Real f5, Real f6, Real f7) {
		v = _mm256_setr_ps((float)f0, (float)f1, (float)f2, (float)f3, 
						   (float)f4, (float)f5, (float)f6, (float)f7);
	}

	Scalarf8(__m256 const & x) {
		v = x;
	}

	Scalarf8 & operator = (__m256 const & x) {
		v = x;
		return *this;
	}

	Scalarf8 & load(float const * p) {
		v = _mm256_loadu_ps(p);
		return *this;
	}
	
	void store(float * p) const {
		_mm256_storeu_ps(p, v);
	}
};

static inline Scalarf8 operator + (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_add_ps(a.v, b.v);
}

static inline Scalarf8 & operator += (Scalarf8 & a, Scalarf8 const & b) {
	a.v = _mm256_add_ps(a.v, b.v);
	return a;
}

static inline Scalarf8 operator - (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_sub_ps(a.v, b.v);
}

static inline Scalarf8 & operator -= (Scalarf8 & a, Scalarf8 const & b) {
	a = a - b;
	return a;
}

static inline Scalarf8 operator * (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_mul_ps(a.v, b.v);
}

static inline Scalarf8 & operator *= (Scalarf8 & a, Scalarf8 const & b) {
	a.v = _mm256_mul_ps(a.v, b.v);
	return a;
}

static inline Scalarf8 operator / (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_div_ps(a.v, b.v);
}

static inline Scalarf8 operator == (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(a.v, b.v, 0);
}

static inline Scalarf8 operator != (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(a.v, b.v, 4);
}

static inline Scalarf8 operator < (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(a.v, b.v, 1);
}

static inline Scalarf8 operator <= (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(a.v, b.v, 2);
}

static inline Scalarf8 operator > (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(b.v, a.v, 1);
}

static inline Scalarf8 operator >= (Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_cmp_ps(b.v, a.v, 2);
}

template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
static inline __m256 constant8f() {
	static const union {
		int     i[8];
		__m256  ymm;
	} u = { { i0,i1,i2,i3,i4,i5,i6,i7 } };
	return u.ymm;
}

static inline Scalarf8 abs(Scalarf8 const & a) {
	__m256 mask = constant8f<0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF>();
	return _mm256_and_ps(a.v, mask);
}

//does the same as for (int i = 0; i < 8; i++) result[i] = c[i] ? a[i] : b[i];
//the elemets in c must be either 0 (false) or 0xFFFFFFFF (true)
static inline Scalarf8 blend(Scalarf8 const & c, Scalarf8 const & a, Scalarf8 const & b) {
	return _mm256_blendv_ps(b.v, a.v, c.v);
}

// ----------------------------------------------------------------------------------------------
//3 dimensional vector of Scalar8f to represent 8 3d vectors
class Vector3f8
{
public:

	Scalarf8 v[3];

	Vector3f8() { v[0] = 0.0; v[1] = 0.0; v[2] = 0.0; }
	Vector3f8(Scalarf8 x, Scalarf8 y, Scalarf8 z) { v[0] = x; v[1] = y; v[2] = z; }
	Vector3f8(Scalarf8 x) { v[0] = v[1] = v[2] = x; }

	inline Scalarf8& operator [] (int i) { return v[i]; }
	inline Scalarf8 operator [] (int i) const { return v[i]; }

	inline Scalarf8& x() { return v[0]; }
	inline Scalarf8& y() { return v[1]; }
	inline Scalarf8& z() { return v[2]; }

	inline Scalarf8 x() const { return v[0]; }
	inline Scalarf8 y() const { return v[1]; }
	inline Scalarf8 z() const { return v[2]; }
	
	inline Scalarf8 dot(const Vector3f8& a) const {
		return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2];
	}

	//dot product
	inline Scalarf8 operator * (const Vector3f8& a) const {
		return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2];
	}

	inline void cross(const Vector3f8& a, const Vector3f8& b) {
		v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
		v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
		v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
	}

	//cross product
	inline const Vector3f8 operator % (const Vector3f8& a) const {
		return Vector3f8(v[1] * a.v[2] - v[2] * a.v[1],
			v[2] * a.v[0] - v[0] * a.v[2],
			v[0] * a.v[1] - v[1] * a.v[0]);
	}

	inline const Vector3f8 operator * (Scalarf8 s) const {
		return Vector3f8(v[0] * s, v[1] * s, v[2] * s);
	}

	inline Vector3f8& operator *= (Scalarf8 s) {
		v[0] *= s;
		v[1] *= s;
		v[2] *= s;
		return *this;
	}

	inline const Vector3f8 operator / (Scalarf8 s) const {
		return Vector3f8(v[0] / s, v[1] / s, v[2] / s);
	}

	inline Vector3f8& operator /= (Scalarf8 s) {
		v[0] = v[0] / s;
		v[1] = v[1] / s;
		v[2] = v[2] / s;
		return *this;
	}

	inline const Vector3f8 operator + (const Vector3f8& a) const {
		return Vector3f8(v[0] + a.v[0], v[1] + a.v[1], v[2] + a.v[2]);
	}

	inline Vector3f8& operator += (const Vector3f8& a) {
		v[0] += a.v[0];
		v[1] += a.v[1];
		v[2] += a.v[2];
		return *this;
	}

	inline const Vector3f8 operator - (const Vector3f8& a) const {
		return Vector3f8(v[0] - a.v[0], v[1] - a.v[1], v[2] - a.v[2]);
	}

	inline Vector3f8& operator -= (const Vector3f8& a) {
		v[0] -= a.v[0];
		v[1] -= a.v[1];
		v[2] -= a.v[2];
		return *this;
	}

	inline const Vector3f8 operator - () const {
		return Vector3f8(Scalarf8(-1.0) * v[0], Scalarf8(-1.0) * v[1], Scalarf8(-1.0) * v[2]);
	}

	inline Scalarf8 lengthSquared() const {
		return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	}

	//does the same as for (int i = 0; i < 8; i++) result[i] = c[i] ? a[i] : b[i];
	//the elemets in c must be either 0 (false) or 0xFFFFFFFF (true)
	static inline Vector3f8 blend(Scalarf8 const & c, Vector3f8 const & a, Vector3f8 const & b) {
		Vector3f8 result;
		result.x() = _mm256_blendv_ps(b.x().v, a.x().v, c.v);
		result.y() = _mm256_blendv_ps(b.y().v, a.y().v, c.v);
		result.z() = _mm256_blendv_ps(b.z().v, a.z().v, c.v);
		return result;
	}
};


// ----------------------------------------------------------------------------------------------
//3x3 dimensional matrix of Scalar8f to represent 8 3x3 matrices
class Matrix3f8
{
public:
	Scalarf8 m[3][3];

	Matrix3f8() {  }

	//constructor to create matrix from 3 column vectors
	Matrix3f8(const Vector3f8& m1, const Vector3f8& m2, const Vector3f8& m3)
	{
		m[0][0] = m1.x();
		m[1][0] = m1.y();
		m[2][0] = m1.z();

		m[0][1] = m2.x();
		m[1][1] = m2.y();
		m[2][1] = m2.z();

		m[0][2] = m3.x();
		m[1][2] = m3.y();
		m[2][2] = m3.z();
	}

	inline Scalarf8& operator()(int i, int j) { return m[i][j]; }

	inline void setCol(int i, const Vector3f8& v)
	{
		m[0][i] = v.x();
		m[1][i] = v.y();
		m[2][i] = v.z();
	}

	inline void setCol(int i, const Scalarf8& x, const Scalarf8& y, const Scalarf8& z)
	{
		m[0][i] = x;
		m[1][i] = y;
		m[2][i] = z;
	}

	inline Vector3f8 operator * (const Vector3f8 &b) const
	{
		Vector3f8 A;

		A.v[0] = m[0][0] * b.v[0] + m[0][1] * b.v[1] + m[0][2] * b.v[2];
		A.v[1] = m[1][0] * b.v[0] + m[1][1] * b.v[1] + m[1][2] * b.v[2];
		A.v[2] = m[2][0] * b.v[0] + m[2][1] * b.v[1] + m[2][2] * b.v[2];

		return A;
	}

	inline Matrix3f8 operator * (const Matrix3f8 &b) const
	{
		Matrix3f8 A;

		A.m[0][0] = m[0][0] * b.m[0][0] + m[0][1] * b.m[1][0] + m[0][2] * b.m[2][0];
		A.m[0][1] = m[0][0] * b.m[0][1] + m[0][1] * b.m[1][1] + m[0][2] * b.m[2][1];
		A.m[0][2] = m[0][0] * b.m[0][2] + m[0][1] * b.m[1][2] + m[0][2] * b.m[2][2];

		A.m[1][0] = m[1][0] * b.m[0][0] + m[1][1] * b.m[1][0] + m[1][2] * b.m[2][0];
		A.m[1][1] = m[1][0] * b.m[0][1] + m[1][1] * b.m[1][1] + m[1][2] * b.m[2][1];
		A.m[1][2] = m[1][0] * b.m[0][2] + m[1][1] * b.m[1][2] + m[1][2] * b.m[2][2];

		A.m[2][0] = m[2][0] * b.m[0][0] + m[2][1] * b.m[1][0] + m[2][2] * b.m[2][0];
		A.m[2][1] = m[2][0] * b.m[0][1] + m[2][1] * b.m[1][1] + m[2][2] * b.m[2][1];
		A.m[2][2] = m[2][0] * b.m[0][2] + m[2][1] * b.m[1][2] + m[2][2] * b.m[2][2];

		return A;
	}

	inline Matrix3f8 transpose() const
	{
		Matrix3f8 A;
		A.m[0][0] = m[0][0]; A.m[0][1] = m[1][0]; A.m[0][2] = m[2][0];
		A.m[1][0] = m[0][1]; A.m[1][1] = m[1][1]; A.m[1][2] = m[2][1];
		A.m[2][0] = m[0][2]; A.m[2][1] = m[1][2]; A.m[2][2] = m[2][2];

		return A;
	}

	inline Scalarf8 determinant() const
	{
		return  m[0][1] * m[1][2] * m[2][0] - m[0][2] * m[1][1] * m[2][0] + m[0][2] * m[1][0] * m[2][1] 
			  - m[0][0] * m[1][2] * m[2][1] - m[0][1] * m[1][0] * m[2][2] + m[0][0] * m[1][1] * m[2][2];
	}

	inline void store(std::vector<Matrix3r>& Mf) const
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				float val[8];
				m[i][j].store(val);
				for (int k = 0; k < 8; k++)
					Mf[k](i, j) = val[k];
			}
		}
	}
};

// ----------------------------------------------------------------------------------------------
//4 dimensional vector of Scalar8f to represent 8 quaternions
class Quaternion8f 
{
public:

	Scalarf8  q[4];

	inline Quaternion8f() { q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0; }

	inline Quaternion8f(Scalarf8 x, Scalarf8 y, Scalarf8 z, Scalarf8 w) {
		q[0] = x; q[1] = y; q[2] = z; q[3] = w;
	}

	inline Quaternion8f(Vector3f8& v) {
		q[0] = v[0]; q[1] = v[1]; q[2] = v[2]; q[3] = 0.0;
	}

	inline Scalarf8 & operator [] (int i) { return q[i]; }
	inline Scalarf8   operator [] (int i) const { return q[i]; }

	inline Scalarf8 & x() { return q[0]; }
	inline Scalarf8 & y() { return q[1]; }
	inline Scalarf8 & z() { return q[2]; }
	inline Scalarf8 & w() { return q[3]; }

	inline Scalarf8 x() const { return q[0]; }
	inline Scalarf8 y() const { return q[1]; }
	inline Scalarf8 z() const { return q[2]; }
	inline Scalarf8 w() const { return q[3]; }

	inline const Quaternion8f operator*(const Quaternion8f& a) const {
		return
			Quaternion8f(q[3] * a.q[0] + q[0] * a.q[3] + q[1] * a.q[2] - q[2] * a.q[1],
				q[3] * a.q[1] - q[0] * a.q[2] + q[1] * a.q[3] + q[2] * a.q[0],
				q[3] * a.q[2] + q[0] * a.q[1] - q[1] * a.q[0] + q[2] * a.q[3],
				q[3] * a.q[3] - q[0] * a.q[0] - q[1] * a.q[1] - q[2] * a.q[2]);
	}

	inline void toRotationMatrix(Matrix3f8& R)
	{
		const Scalarf8 tx = Scalarf8(2.0) * q[0];
		const Scalarf8 ty = Scalarf8(2.0) * q[1];
		const Scalarf8 tz = Scalarf8(2.0) * q[2];
		const Scalarf8 twx = tx*q[3];
		const Scalarf8 twy = ty*q[3];
		const Scalarf8 twz = tz*q[3];
		const Scalarf8 txx = tx*q[0];
		const Scalarf8 txy = ty*q[0];
		const Scalarf8 txz = tz*q[0];
		const Scalarf8 tyy = ty*q[1];
		const Scalarf8 tyz = tz*q[1];
		const Scalarf8 tzz = tz*q[2];

	    R.m[0][0] = Scalarf8(1.0) - (tyy + tzz);
		R.m[0][1] = txy - twz;
		R.m[0][2] = txz + twy;
		R.m[1][0] = txy + twz;
		R.m[1][1] = Scalarf8(1.0) - (txx + tzz);
		R.m[1][2] = tyz - twx;
		R.m[2][0] = txz - twy;
		R.m[2][1] = tyz + twx;
		R.m[2][2] = Scalarf8(1.0) - (txx + tyy);
	}

	inline void toRotationMatrix(Vector3f8& R1, Vector3f8& R2, Vector3f8& R3)
	{
		const Scalarf8 tx = Scalarf8(2.0) * q[0];
		const Scalarf8 ty = Scalarf8(2.0) * q[1];
		const Scalarf8 tz = Scalarf8(2.0) * q[2];
		const Scalarf8 twx = tx*q[3];
		const Scalarf8 twy = ty*q[3];
		const Scalarf8 twz = tz*q[3];
		const Scalarf8 txx = tx*q[0];
		const Scalarf8 txy = ty*q[0];
		const Scalarf8 txz = tz*q[0];
		const Scalarf8 tyy = ty*q[1];
		const Scalarf8 tyz = tz*q[1];
		const Scalarf8 tzz = tz*q[2];

		R1[0] = Scalarf8(1.0) - (tyy + tzz);
		R2[0] = txy - twz;
		R3[0] = txz + twy;
		R1[1] = txy + twz;
		R2[1] = Scalarf8(1.0) - (txx + tzz);
		R3[1] = tyz - twx;
		R1[2] = txz - twy;
		R2[2] = tyz + twx;
		R3[2] = Scalarf8(1.0) - (txx + tyy);
	}

	inline void store(std::vector<Quaternionr>& qf) const
	{
		float x[8], y[8], z[8], w[8];
		q[0].store(x);
		q[1].store(y);
		q[2].store(z);
		q[3].store(w);

		for (int i = 0; i < 8; i++)
		{
			qf[i].x() = x[i];
			qf[i].y() = y[i];
			qf[i].z() = z[i];
			qf[i].w() = w[i];
		}
	}

	inline void set(const std::vector<Quaternionr>& qf)
	{
		float x[8], y[8], z[8], w[8];
		for(int i=0; i<8; i++)
		{
			x[i] = static_cast<float>(qf[i].x());
			y[i] = static_cast<float>(qf[i].y());
			z[i] = static_cast<float>(qf[i].z());
			w[i] = static_cast<float>(qf[i].w());
		}
		Scalarf8 s;
		s.load(x);
		q[0] = s;
		s.load(y);
		q[1] = s; 
		s.load(z);
		q[2] = s; 
		s.load(w);
		q[3] = s;
	}
};

// ----------------------------------------------------------------------------------------------
//alligned allocator so that vectorized types can be used in std containers
//from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
template <typename T, std::size_t N = 32>
class AlignmentAllocator {
public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	typedef T * pointer;
	typedef const T * const_pointer;

	typedef T & reference;
	typedef const T & const_reference;

public:
	inline AlignmentAllocator() throw () { }

	template <typename T2>
	inline AlignmentAllocator(const AlignmentAllocator<T2, N> &) throw () { }

	inline ~AlignmentAllocator() throw () { }

	inline pointer adress(reference r) {
		return &r;
	}

	inline const_pointer adress(const_reference r) const {
		return &r;
	}

	inline pointer allocate(size_type n) {
#ifdef _WIN32
		return (pointer)_aligned_malloc(n * sizeof(value_type), N);
#elif __linux__
		// NB! Argument order is opposite from MSVC/Windows
		return (pointer) aligned_alloc(N, n * sizeof(value_type));
#else
#error "Unknown platform"
#endif
	}

	inline void deallocate(pointer p, size_type) {
#ifdef _WIN32
		_aligned_free(p);
#elif __linux__
		free(p);
#else
#error "Unknown platform"
#endif
	}

	inline void construct(pointer p, const value_type & wert) {
		new (p) value_type(wert);
	}

	inline void destroy(pointer p) {
		p->~value_type();
	}

	inline size_type max_size() const throw () {
		return size_type(-1) / sizeof(value_type);
	}

	template <typename T2>
	struct rebind {
		typedef AlignmentAllocator<T2, N> other;
	};

	bool operator!=(const AlignmentAllocator<T, N>& other) const {
		return !(*this == other);
	}

	// Returns true if and only if storage allocated from *this
	// can be deallocated from other, and vice versa.
	// Always returns true for stateless allocators.
	bool operator==(const AlignmentAllocator<T, N>& other) const {
		return true;
	}
};
#endif