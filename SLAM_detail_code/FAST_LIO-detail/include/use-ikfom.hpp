#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

/*
ikfom状态的定义:
位置：3维向量
旋转：SO3上的旋转，表示旋转
IMU到Lidar的旋转：SO3上的旋转，表示IMU到Lidar的旋转
IMU到Lidar的平移：3维向量，表示IMU到Lidar的平移
速度：3维向量，表示速度
陀螺仪偏置：3维向量，表示陀螺仪偏置
加速度计偏置：3维向量，表示加速度计偏置
重力：流形S2上的向量，2维，表示重力
*/
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((S2, grav))
);

// ikfom输入的定义:
// 加速度：3维向量，表示加速度
// 陀螺仪：3维向量，表示陀螺仪
MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

// ikfom过程噪声的定义:
// ng:陀螺仪测量白噪声
// na:加速度计测量白噪声
// nbg:陀螺仪偏置的随机游走噪声
// nba:加速度计偏置的随机游走噪声
MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

// 初始化过程噪声协方差矩阵
MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

// 计算状态转移矩阵f
//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);  // omega = gyro - bg
	vect3 a_inertial = s.rot * (in.acc-s.ba);  // a_inertial = R * (acc - ba)
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];  // 速度
		res(i + 3) =  omega[i];  // 角速度
		// res(i + 6) = s.offset_T_L_I[i];  // 跳过了lidar到imu的tf
		// res(i + 9) = s.offset_R_L_I[i];
		res(i + 12) = a_inertial[i] + s.grav[i];  // 加速度
		// res(i + 15) = bg_dot;
		// res(i + 18) = ba_dot;
		// res(i + 21) = grav_dot;
	}
	return res;
}

// 计算状态转移矩阵f的雅克比矩阵
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();  // 速度对位置的雅克比矩阵
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);  // acc_ = acc - ba
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);  // omega = gyro - bg
	// 连续时间IMU模型 v_dot = R * (acc - ba) + g
	// d(v_dot)/d(theta) = -R * hat(acc)  // v_dot对姿态误差theta的偏导
	// d(v_dot)/d(ba) = -R                // v_dot对加速度计偏置ba的偏导
 	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 21);  // 对于S2上的向量g，计算d(g)/d(phi)
	cov.template block<3, 2>(12, 21) =  grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();  // 角速度对陀螺仪偏置bg的偏导
	return cov;
}

// 计算状态转移矩阵f的噪声雅克比矩阵
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();  // 加速度对加速度计测量白噪声na的偏导
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();  // 角速度对陀螺仪测量白噪声ng的偏导
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();  // 陀螺仪偏置对陀螺仪偏置随机游走噪声nbg的偏导
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();  // 加速度计偏置对加速度计偏置随机游走噪声nba的偏导
	return cov;
}

// 将SO3上的旋转转换为欧拉角
vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif