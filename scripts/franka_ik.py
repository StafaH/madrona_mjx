import numpy as np
import jax
import jax.numpy as jp
from brax import math

PI_2 = jp.pi/2
PI_4 = jp.pi/4

def mat_from_dh_revolute(theta, alpha, a, d, offset):
  ca = jp.cos(alpha)
  sa = jp.sin(alpha)
  th = 0.0
  th = theta + offset
  ct = jp.cos(th)
  st = jp.sin(th)
  mat = jp.asarray([
      [ct, -st, 0, a],
      [st * ca, ct * ca, -sa, -d * sa],
      [st * sa, ct * sa, ca, d * ca],
      [0, 0, 0, 1]], dtype=jp.float32)
  return mat

def compute_franka_fk(joint_pos):
  
  l1 = 0.333
  l3 = 0.316
  l4 = 0.0825
  l5a = -0.0825
  l5d = 0.384
  l7 = 0.088
  flange = 0.107
  
  # To flange is 0.088 and to hand tip is 0.2104
  # l7 = 0.2104
  # l7 = 0.088

  t_7_0 = jp.identity(4)
  t_1_0 = mat_from_dh_revolute(joint_pos[0], 0, 0, l1, 0)
  t_2_1 = mat_from_dh_revolute(joint_pos[1], -PI_2, 0, 0, 0)
  t_3_2 = mat_from_dh_revolute(joint_pos[2], PI_2, 0, l3, 0)
  t_4_3 = mat_from_dh_revolute(joint_pos[3], PI_2, l4, 0, 0)
  t_5_4 = mat_from_dh_revolute(joint_pos[4], -PI_2, l5a, l5d, 0)
  t_6_5 = mat_from_dh_revolute(joint_pos[5], PI_2, 0, 0, 0)
  t_7_6 = mat_from_dh_revolute(joint_pos[6], PI_2, l7, 0, 0)
  t_f_7 = mat_from_dh_revolute(0, 0, 0, flange, 0)
  
  t_7_0 = jp.matmul(t_7_0, t_1_0)
  t_7_0 = jp.matmul(t_7_0, t_2_1)
  t_7_0 = jp.matmul(t_7_0, t_3_2)
  t_7_0 = jp.matmul(t_7_0, t_4_3)
  t_7_0 = jp.matmul(t_7_0, t_5_4)
  t_7_0 = jp.matmul(t_7_0, t_6_5)
  t_7_0 = jp.matmul(t_7_0, t_7_6)
  t_7_0 = jp.matmul(t_7_0, t_f_7)

  return t_7_0


def franka_compute_ik(t_7_0, q7, q_actual):
  # Jax adaptation of C++ franka ik solution from https://github.com/ffall007/franka_analytical_ik/blob/main/franka_ik_He.hpp
  # Write the above C++ code in pure jax numpy
  
  q_nan = jp.array([jp.nan] * 7)
  q = jp.array([0.0] * 7, dtype=jp.float32)

  d1 = 0.3330
  d3 = 0.3160
  d5 = 0.3840
  # d7e = 0.2104
  #alt
  d7e = 0.107
  a4 = 0.0825
  a7 = 0.0880

  LL24 = 0.10666225
  LL46 = 0.15426225
  L24 = 0.326591870689
  L46 = 0.392762332715

  thetaH46 = 1.35916951803
  theta342 = 1.31542071191
  theta46H = 0.211626808766

  q_min = jp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
  q_max = jp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

  q = q.at[6].set(q7)

  c1_a = jp.cos(q_actual[0])
  s1_a = jp.sin(q_actual[0])
  c2_a = jp.cos(q_actual[1])
  s2_a = jp.sin(q_actual[1])
  c3_a = jp.cos(q_actual[2])
  s3_a = jp.sin(q_actual[2])
  c4_a = jp.cos(q_actual[3])
  s4_a = jp.sin(q_actual[3])
  c5_a = jp.cos(q_actual[4])
  s5_a = jp.sin(q_actual[4])
  c6_a = jp.cos(q_actual[5])
  s6_a = jp.sin(q_actual[5])

  As_a = []

  As_a.append(jp.array([
    [c1_a, -s1_a, 0.0, 0.0],
    [s1_a, c1_a, 0.0, 0.0],
    [0.0, 0.0, 1.0, d1],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [c2_a, -s2_a, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [-s2_a, -c2_a, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [c3_a, -s3_a, 0.0, 0.0],
    [0.0, 0.0, -1.0, -d3],
    [s3_a, c3_a, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [c4_a, -s4_a, 0.0, a4],
    [0.0, 0.0, -1.0, 0.0],
    [s4_a, c4_a, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [1.0, 0.0, 0.0, -a4],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [c5_a, -s5_a, 0.0, 0.0],
    [0.0, 0.0, 1.0, d5],
    [-s5_a, -c5_a, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  As_a.append(jp.array([
    [c6_a, -s6_a, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [s6_a, c6_a, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]))

  Ts_a = []
  Ts_a.append(As_a[0])
  for j in range(1, 7):
    Ts_a.append(jp.matmul(Ts_a[j - 1], As_a[j]))

  # identify q6 case
  # Double check this
  V62_a = Ts_a[1][:3, 3] - Ts_a[6][:3, 3]
  V6H_a = Ts_a[4][:3, 3] - Ts_a[6][:3, 3]
  Z6_a = Ts_a[6][:3, 2]
  is_case6_0 = jp.sum(jp.matmul(jp.cross(V6H_a, V62_a), Z6_a)) <= 0

  # identify q1 case
  is_case1_1 = q_actual[1] < 0

  # IK: compute p_6
  R_EE = t_7_0[:3, :3]
  z_EE = t_7_0[:3, 2]
  p_EE = t_7_0[:3, 3]
  p_7 = p_EE - (d7e*z_EE)

  x_EE_6 = jp.array([jp.cos(q7 - PI_4), -jp.sin(q7 - PI_4), 0.0])
  x_6 = jp.matmul(R_EE, x_EE_6)
  x_6 /= jp.linalg.norm(x_6)
  p_6 = p_7 - a7*x_6

  # IK: compute q4
  p_2 = jp.array([0.0, 0.0, d1])
  V26 = p_6 - p_2

  LL26 = jp.sum(V26 * V26)
  L26 = jp.sqrt(LL26)

  # if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
  #   return q_nan
  
  theta246 = jp.arccos((LL24 + LL46 - LL26) / 2.0 / L24 / L46)
  q = q.at[3].set(theta246 + thetaH46 + theta342 - 2.0*jp.pi)

  # if q[3] <= q_min[3] or q[3] >= q_max[3]:
  #   return q_nan
  
  # IK: compute q6
  theta462 = jp.arccos((LL26 + LL46 - LL24) / 2.0 / L26 / L46)
  theta26H = theta46H + theta462
  D26 = -L26 * jp.cos(theta26H)

  Z_6 = jp.cross(z_EE, x_6)
  Y_6 = jp.cross(Z_6, x_6)
  R_6 = jp.column_stack((x_6, Y_6 / jp.linalg.norm(Y_6), Z_6 / jp.linalg.norm(Z_6)))
  V_6_62 = jp.matmul(R_6.T, -V26)

  Phi6 = jp.arctan2(V_6_62[1], V_6_62[0])
  Theta6 = jp.arcsin(D26 / jp.sqrt((V_6_62[0] * V_6_62[0]) + (V_6_62[1] * V_6_62[1])))

  # if is_case6_0:
  #   q = q.at[5].set(jp.pi - Theta6 - Phi6)
  # else:
  #   q = q.at[5].set(Theta6 - Phi6)
  q = jp.where(
    is_case6_0,
    q.at[5].set(jp.pi - Theta6 - Phi6),
    q.at[5].set(Theta6 - Phi6))
  
  # if q[5] <= q_min[5]:
  #   q = q.at[5].set(q[5] + 2.0 * jp.pi)
  # elif q[5] >= q_max[5]:
  #   q = q.at[5].set(q[5] - 2.0 * jp.pi)
  q = jp.where(
    q[5] <= q_min[5],
    q.at[5].set(q[5] + 2.0 * jp.pi),
    q)
  q = jp.where(
    q[5] >= q_max[5],
    q.at[5].set(q[5] - 2.0 * jp.pi),
    q)
  
  # if q[5] <= q_min[5] or q[5] >= q_max[5]:
  #   return q_nan

  # IK: compute q1 & q2
  thetaP26 = 3.0 * jp.pi / 2 - theta462 - theta246 - theta342
  thetaP = jp.pi - thetaP26 - theta26H
  LP6 = L26 * jp.sin(thetaP26) / jp.sin(thetaP)

  z_6_5 = jp.array([jp.sin(q[5]), jp.cos(q[5]), 0.0])
  z_5 = jp.matmul(R_6, z_6_5)
  V2P = p_6 - LP6 * z_5 - p_2

  L2P = jp.linalg.norm(V2P)

  # if jp.abs(V2P[2] / L2P) > 0.999:
  #   q = q.at[0].set(q_actual[0])
  #   q = q.at[1].set(0.0)
  # else:
  #   q = q.at[0].set(jp.arctan2(V2P[1], V2P[0]))
  #   q = q.at[1].set(jp.arccos(V2P[2] / L2P))
  #   if is_case1_1:
  #     if q[0] < 0.0:
  #       q = q.at[0].set(q[0] + jp.pi)
  #     else:
  #       q = q.at[0].set(q[0] - jp.pi)
  #     q = q.at[1].set(-q[1])
  # Replace with jp.where
  V2P_L2P = jp.abs(V2P[2] / L2P)
  greater_than_1 = jp.where(V2P_L2P > 0.999, True, False)

  q = jp.where(
    greater_than_1,
    q.at[0].set(q_actual[0]),
    q.at[0].set(jp.arctan2(V2P[1], V2P[0]))
  )
  q = jp.where(
    greater_than_1,
    q.at[1].set(0.0),
    q.at[1].set(jp.arccos(V2P[2] / L2P))
  )

  is_q_less_than_0 = jp.where(q[0] < 0.0, True, False)
  q = jp.where(
    is_case1_1,
    jp.where(
      is_q_less_than_0,
      q.at[0].set(q[0] + jp.pi),
      q.at[0].set(q[0] - jp.pi)
    ),
    q
  )
  q = jp.where(
    not greater_than_1 and is_case1_1,
    q.at[1].set(-q[1]),
    q
  )
  
  # if q[0] <= q_min[0] or q[0] >= q_max[0] or q[1] <= q_min[1] or q[1] >= q_max[1]:
  #   return q_nan

  # IK: compute q3
  z_3 = V2P / L2P
  Y_3 = -jp.cross(V26, V2P)
  y_3 = Y_3 / jp.linalg.norm(Y_3)
  x_3 = jp.cross(y_3, z_3)
  c1 = jp.cos(q[0])
  s1 = jp.sin(q[0])
  R_1 = jp.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])

  c2 = jp.cos(q[1])
  s2 = jp.sin(q[1])
  R_1_2 = jp.array([[c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
  R_2 = jp.matmul(R_1, R_1_2)
  x_2_3 = jp.matmul(R_2.T, x_3)
  q = q.at[2].set(jp.arctan2(x_2_3[2], x_2_3[0]))

  # if q[2] <= q_min[2] or q[2] >= q_max[2]:
  #   return q_nan
  
  # IK: compute q4
  VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5
  c6 = jp.cos(q[5])
  s6 = jp.sin(q[5])
  R_5_6 = jp.array([[c6, -s6, 0.0], [0.0, 0.0, -1.0], [s6, c6, 0.0]])
  R_5 = jp.matmul(R_6, R_5_6.T)
  V_5_H4 = jp.matmul(R_5.T, VH4)

  q = q.at[4].set(-jp.arctan2(V_5_H4[1], V_5_H4[0]))
  # if q[4] <= q_min[4] or q[4] >= q_max[4]:
  #   return q_nan
  
  return q


if __name__ == '__main__':
  # joint_pos = jp.array([-0.261, -0.048, -0.166, -2.023, 0.024, 1.994, 0.0])
  # joint_pos = jp.array([2.70474, 0.25356, -2.71034, -2.33938,
  #                       0.084890, 2.11775, 0.54595])
  home = jp.array([0, 0.3, 0, -1.57079, 0, 2.0, 0.785398])
  t_7_0 = compute_franka_fk(home)

  for row in t_7_0:
    for elem in row:
      print(elem, end=', ')

  print('home end effector assuming no hand')
  print(t_7_0)

  # q_actual = jp.array([0, 0.3, 0, -1.57079, 0, 2.0, 0.785398])
  q_actual =  jp.array([0, 0.3, 0, -1.57079, 0, 2.0, 0.785398])

  # Add 45 degree offset to remote hand offset
  q7 = q_actual[6] + jp.pi/4
  q_actual = q_actual.at[6].set(q7)
  q = franka_compute_ik(t_7_0, q7, q_actual)

  print(q)
