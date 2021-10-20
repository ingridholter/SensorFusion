import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from config import DEBUG
from cross_matrix import get_cross_matrix

import solution
from solution import quaternion


@dataclass
class RotationQuaterion:
    """Class representing a rotation quaternion (norm = 1). Has some useful
    methods for converting between rotation representations.

    Hint: You can implement all methods yourself, or use scipys Rotation class.
    scipys Rotation uses the xyzw notation for quats while the book uses wxyz
    (this i really annoying, I know).

    Args:
        real_part (float): eta (n) in the book, w in scipy notation
        vec_part (ndarray[3]): epsilon in the book, (x,y,z) in scipy notation
    """
    real_part: float
    vec_part: 'ndarray[3]'

    def __post_init__(self):
        if DEBUG:
            assert len(self.vec_part) == 3

        norm = self.real_part**2 + sum(self.vec_part**2)
        if not np.allclose(norm, 1):
            self.real_part /= norm
            self.vec_part /= norm

        if self.real_part < 0:
            self.real_part *= -1
            self.vec_part *= -1

    def multiply(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        """Multiply two rotation quaternions
        Hint: see (10.33)

        As __matmul__ is implemented for this class, you can use:
        q1@q2 which is equivalent to q1.multiply(q2)

        Args:
            other (RotationQuaternion): the other quaternion  
        Returns:
            quaternion_product (RotationQuaternion): the product
        """

        quaternion_product_sol = solution.quaternion.RotationQuaterion.multiply(
            self, other)
        # self_matrix2_row1 = np.concatenate([0,-self.vec_part], axis=None).reshape(1,4)
        # self_matrix2_row24 = np.concatenate([self.vec_part.reshape(3,1),get_cross_matrix(self.vec_part)], axis = 1)
        # self_matrix2 = np. concatenate([self_matrix2_row1,self_matrix2_row24],axis=0)
        # self_matrix = self.real_part*np.eye(4) + self_matrix2
        # quaternion_product = self_matrix@np.concatenate([other.real_part, other.vec_part], axis = None).reshape(4,1)
        #return RotationQuaterion(real_part=quaternion_product[0,0],vec_part=quaternion_product[0,1:].reshape(3))
        return quaternion_product_sol

    def conjugate(self) -> 'RotationQuaterion':
        """Get the conjugate of the RotationQuaternion"""

        #conj_sol = solution.quaternion.RotationQuaterion.conjugate(self)
        conj = RotationQuaterion(real_part=self.real_part,vec_part=-self.vec_part)
        return conj

    def as_rotmat(self) -> 'ndarray[3,3]':
        """Get the rotation matrix representation of self

        Returns:
            R (ndarray[3,3]): rotation matrix
        """
        quat = np.concatenate([self.vec_part, self.real_part], axis=None)
        r = Rotation.from_quat(quat)
        R = r.as_matrix()
        #R_sol = solution.quaternion.RotationQuaterion.as_rotmat(self)
        return R

    @property
    def R(self) -> 'ndarray[3,3]':
        return self.as_rotmat()

    def as_euler(self) -> 'ndarray[3]':
        """Get the euler angle representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """
        quat = np.concatenate([self.vec_part, self.real_part], axis=None)
        r = Rotation.from_quat(quat)
        euler= r.as_euler('xyz')

        #euler_sol = solution.quaternion.RotationQuaterion.as_euler(self)
        
        return euler

    def as_avec(self) -> 'ndarray[3]':
        """Get the angles vector representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """

        quat = np.concatenate([self.vec_part, self.real_part], axis=None)
        r = Rotation.from_quat(quat)
        avec = r.as_rotvec()
        #avec_sol = solution.quaternion.RotationQuaterion.as_avec(self)

        return avec

    @staticmethod
    def from_euler(euler: 'ndarray[3]') -> 'RotationQuaterion':
        """Get a rotation quaternion from euler angles
        usage: rquat = RotationQuaterion.from_euler(euler)

        Args:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)

        Returns:
            rquat (RotationQuaternion): the rotation quaternion
        """
        scipy_quat = Rotation.from_euler('xyz', euler).as_quat()
        rquat = RotationQuaterion(scipy_quat[3], scipy_quat[:3])
        return rquat

    def _as_scipy_quat(self):
        """If you're using scipys Rotation class, this can be handy"""
        return np.append(self.vec_part, self.real_part)

    def __iter__(self):
        return iter([self.real_part, self.vec_part])

    def __matmul__(self, other) -> 'RotationQuaterion':
        """Lets u use the @ operator, q1@q2 == q1.multiply(q2)"""
        return self.multiply(other)
