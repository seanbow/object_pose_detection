#ifndef VISO_MATH_H_
#define VISO_MATH_H_

#include <Eigen/Dense>

inline Eigen::Matrix3d skewsymm(const Eigen::Vector3d& a);
inline Eigen::Matrix4d quat_L_matrix(const Eigen::Vector4d& q);

inline Eigen::Vector4d quat_vec_mul(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2);

Eigen::Vector4d quat_mul(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2);

inline Eigen::Vector4d quat_inv(const Eigen::Vector4d& q);

inline Eigen::MatrixXd xi_quat_mat(const Eigen::Vector4d& q);

inline Eigen::MatrixXd relpose_difference_jacobian_1(const Eigen::Matrix4d& xj, 
												     const Eigen::Matrix4d& xi);

inline Eigen::Matrix3d quat2rot(const Eigen::Vector4d& q);

//
//  rot2quat
//% converts a rotational matrix to a unit quaternion, according to JPL
//% procedure (Breckenridge Memo)
inline Eigen::Vector4d rot2quat(const Eigen::MatrixXd &rot);

inline Eigen::Matrix3d skewsymm(const Eigen::Vector3d& a) {
    Eigen::Matrix3d S;
    S <<   0,  -a(2),  a(1),
          a(2),  0  , -a(0), 
         -a(1), a(0),   0  ;
    return S;
}

inline Eigen::Matrix4d quat_L_matrix(const Eigen::Vector4d& q) {
    Eigen::Matrix4d L;

    L.block<3,3>(0,0) = q(3)*Eigen::Matrix3d::Identity() - skewsymm(q.head<3>());
    L.block<1,3>(3,0) = -q.head<3>().transpose();
    L.block<3,1>(0,3) = q.head<3>();
    L(3,3) = q(3);

    return L;
}

inline Eigen::Vector4d quat_vec_mul(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
    return quat_L_matrix(q1) * q2;
}

Eigen::Vector4d quat_mul(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
    Eigen::Vector4d q = quat_vec_mul(q1, q2);

    if ( q(3) < 0 ) {
        q = -q;
    }

    return q;
}

inline Eigen::Vector4d quat_inv(const Eigen::Vector4d& q) {
	Eigen::Vector4d qi = q;
	qi.head<3>() = -qi.head<3>();
	return qi;
}

inline Eigen::MatrixXd xi_quat_mat(const Eigen::Vector4d& q) {
	Eigen::MatrixXd Xi = Eigen::MatrixXd::Zero(4,3);

	Xi <<  q(3), -q(2),  q(1),
	       q(2),  q(3), -q(0),
	      -q(1),  q(0),  q(3),
	      -q(0), -q(1), -q(2);

	return Xi;
}

inline Eigen::MatrixXd psi_quat_mat(const Eigen::Vector4d& q) {
	Eigen::MatrixXd Psi = Eigen::MatrixXd::Zero(4,3);

	Psi <<  q(3),  q(2), -q(1),
	       -q(2),  q(3),  q(0),
	        q(1), -q(0),  q(3),
	       -q(0), -q(1), -q(2);

	return Psi;
}

inline Eigen::MatrixXd relpose_difference_jacobian_1(const Eigen::Matrix4d& xj, const Eigen::Matrix4d& xi){
	Eigen::Vector3d pi = xi.block<3,1>(0,3);
	Eigen::Vector3d pj = xj.block<3,1>(0,3);
	Eigen::Vector4d qi = rot2quat( xi.block<3,3>(0,0) );
	Eigen::Vector4d qj = rot2quat( xj.block<3,3>(0,0) );

	Eigen::MatrixXd Jj = Eigen::MatrixXd::Zero(6,6);

	Jj.block<3,3>(0,0) = xi.block<3,3>(0,0);

	auto Xi = xi_quat_mat(quat_mul(qj, quat_inv(qi)));

	Eigen::Matrix<double, 3, 4> premul = Eigen::Matrix<double, 3, 4>::Zero();
	premul.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

	Jj.block<3,3>(3,3) = premul * Xi;

	return Jj;
}

inline Eigen::MatrixXd relpose_difference_jacobian_2(const Eigen::Matrix4d& xj, const Eigen::Matrix4d& xi){
	Eigen::Vector3d pi = xi.block<3,1>(0,3);
	Eigen::Vector3d pj = xj.block<3,1>(0,3);
	Eigen::Vector4d qi = rot2quat( xi.block<3,3>(0,0) );
	Eigen::Vector4d qj = rot2quat( xj.block<3,3>(0,0) );

	Eigen::MatrixXd Ji = Eigen::MatrixXd::Zero(6,6);

	Ji.block<3,3>(0,0) = -xi.block<3,3>(0,0);
	Ji.block<3,3>(0,3) = 2*skewsymm( xi.block<3,3>(0,0) * (pj - pi) );

	auto Psi = psi_quat_mat( quat_mul(qj, quat_inv(qi)) );

	Eigen::Matrix<double, 3, 4> premul = Eigen::Matrix<double, 3, 4>::Zero();
	premul.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

	Ji.block<3,3>(3,3) = -premul * Psi;

	return Ji;
}

inline Eigen::Matrix3d quat2rot(const Eigen::Vector4d& q) {
    auto skewed = skewsymm(q.head<3>());
    return Eigen::Matrix3d::Identity() - 2*q(3)*skewed + 2*skewed*skewed;
}

//
//  rot2quat
//% converts a rotational matrix to a unit quaternion, according to JPL
//% procedure (Breckenridge Memo)
inline Eigen::Vector4d rot2quat(const Eigen::MatrixXd &rot)  {
    Eigen::Vector4d q;
    double T = rot.trace();
     if ((rot(0,0)>T) && (rot(0,0)>rot(1,1)) && (rot(0,0)>rot(2,2))) {
        q(0) = sqrt((1+(2*rot(0,0))-T)/4);
        q(1) = (1/(4*q(0))) * (rot(0,1)+rot(1,0));
        q(2) = (1/(4*q(0))) * (rot(0,2)+rot(2,0));
        q(3) = (1/(4*q(0))) * (rot(1,2)-rot(2,1));

    } else if ((rot(1,1)>T) && (rot(1,1)>rot(0,0)) && (rot(1,1)>rot(2,2))) {
        q(1) = sqrt((1+(2*rot(1,1))-T)/4);
        q(0) = (1/(4*q(1))) * (rot(0,1)+rot(1,0));
        q(2) = (1/(4*q(1))) * (rot(1,2)+rot(2,1));
        q(3) = (1/(4*q(1))) * (rot(2,0)-rot(0,2));
    } else if ((rot(2,2)>T) && (rot(2,2)>rot(0,0)) && (rot(2,2)>rot(1,1))) {
        q(2) = sqrt((1+(2*rot(2,2))-T)/4);
        q(0) = (1/(4*q(2))) * (rot(0,2)+rot(2,0));
        q(1) = (1/(4*q(2))) * (rot(1,2)+rot(2,1));
        q(3) = (1/(4*q(2))) * (rot(0,1)-rot(1,0));
    } else {

        q(3) = sqrt((1+T)/4);
        q(0) = (1/(4*q(3))) * (rot(1,2)-rot(2,1));
        q(1) = (1/(4*q(3))) * (rot(2,0)-rot(0,2));
        q(2) = (1/(4*q(3))) * (rot(0,1)-rot(1,0));
    }
    if  (q(3)<0) {
        q = -q;
    }
    q = q/q.norm();
    return q;

}

#endif