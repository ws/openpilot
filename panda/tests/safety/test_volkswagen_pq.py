#!/usr/bin/env python3
import unittest
import numpy as np
from panda import Panda
from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import test_relay_malfunction, make_msg, \
  test_manually_enable_controls_allowed, test_spam_can_buses, MAX_WRONG_COUNTERS

MAX_RATE_UP = 4
MAX_RATE_DOWN = 10
MAX_STEER = 300
MAX_RT_DELTA = 75
RT_INTERVAL = 250000

DRIVER_TORQUE_ALLOWANCE = 80
DRIVER_TORQUE_FACTOR = 3

MSG_EPS_1 = 0x0D0       # RX from EPS, for steering angle and driver steering torque
MSG_HCA_1 = 0x0D2       # TX by OP, Heading Control Assist steering torque
MSG_MOTOR_1 = 0x280     # RX from ECU, for driver throttle input
MSG_MOTOR_2 = 0x288     # RX from ECU, for CC state and brake switch state
MSG_GRA_NEU = 0x38A     # TX by OP, ACC control buttons for cancel/resume
MSG_BREMSE_3 = 0x4A0    # RX from ABS, for wheel speeds
MSG_LDW_1 = 0x5BE       # TX by OP, Lane line recognition and text alerts

# Transmit of GRA_Neu is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
TX_MSGS = [[MSG_HCA_1, 0], [MSG_GRA_NEU, 0], [MSG_GRA_NEU, 2], [MSG_LDW_1, 0]]

def sign(a):
  if a > 0:
    return 1
  else:
    return -1

def volkswagen_pq_checksum(msg, addr, len_msg):
  msg_bytes = msg.RDLR.to_bytes(4, 'little') + msg.RDHR.to_bytes(4, 'little')
  msg_bytes = msg_bytes[1:len_msg]

  checksum = 0
  for i in msg_bytes:
    checksum ^= i
  return checksum

class TestVolkswagenPqSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_VOLKSWAGEN_PQ, 0)
    cls.safety.init_tests_volkswagen()
    cls.cnt_eps_1 = 0
    cls.cnt_hca_1 = 0

  def _set_prev_torque(self, t):
    self.safety.set_volkswagen_desired_torque_last(t)
    self.safety.set_volkswagen_rt_torque_last(t)

  # Wheel speeds
  def _bremse_3_msg(self, speed):
    wheel_speed_scaled = int(speed / 0.01)
    to_send = make_msg(0, MSG_BREMSE_3)
    to_send[0].RDLR = (wheel_speed_scaled | (wheel_speed_scaled << 16)) << 1
    to_send[0].RDHR = (wheel_speed_scaled | (wheel_speed_scaled << 16)) << 1
    return to_send

  # Driver steering input torque
  def _eps_1_msg(self, torque):
    to_send = make_msg(0, MSG_EPS_1)
    t = abs(torque)
    to_send[0].RDLR = ((t & 0x3FF) << 16)
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 26
    to_send[0].RDLR |= (self.cnt_eps_1 % 16) << 8
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_EPS_1, 8)
    self.cnt_eps_1 += 1
    return to_send

  # openpilot steering output torque
  def _hca_1_msg(self, torque):
    to_send = make_msg(0, MSG_HCA_1)
    t = abs(torque)
    to_send[0].RDLR = (t & 0x7FFF) << 16
    if torque < 0:
      to_send[0].RDLR |= 0x1 << 31
    to_send[0].RDLR |= (self.cnt_hca_1 % 16) << 8
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_HCA_1, 8)
    self.cnt_hca_1 += 1
    return to_send

  # ACC engagement and brake light switch status
  def _motor_2_msg(self, acc, brake):
    to_send = make_msg(0, MSG_MOTOR_2)
    to_send[0].RDLR = (0x1 << 16) if brake else 0
    to_send[0].RDLR |= (acc & 0x3) << 22
    return to_send

  # Driver throttle input
  def _motor_1_msg(self, gas):
    to_send = make_msg(0, MSG_MOTOR_1)
    to_send[0].RDHR = (int(gas / 0.4) & 0xFF) << 8
    return to_send

  # Cruise control buttons
  def _gra_neu_msg(self, bit):
    to_send = make_msg(2, MSG_GRA_NEU)
    to_send[0].RDLR = 1 << bit
    to_send[0].RDLR |= volkswagen_pq_checksum(to_send[0], MSG_GRA_NEU, 8)
    return to_send

  def test_spam_can_buses(self):
    test_spam_can_buses(self, TX_MSGS)

  def test_relay_malfunction(self):
    test_relay_malfunction(self, MSG_HCA_1)

  def test_prev_gas(self):
    for g in range(0, 256):
      self.safety.safety_rx_hook(self._motor_1_msg(g))
      self.assertEqual(g, self.safety.get_volkswagen_gas_prev())

  def test_default_controls_not_allowed(self):
    self.assertFalse(self.safety.get_controls_allowed())

  def test_enable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(0)
    self.safety.safety_rx_hook(self._motor_2_msg(2, False))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_disable_control_allowed_from_cruise(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._motor_2_msg(0, False))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_sample_speed(self):
    self.assertEqual(0, self.safety.get_volkswagen_moving())
    self.safety.safety_rx_hook(self._bremse_3_msg(100))
    self.assertEqual(1, self.safety.get_volkswagen_moving())

  def test_prev_brake(self):
    self.assertFalse(self.safety.get_volkswagen_brake_pressed_prev())
    self.safety.safety_rx_hook(self._motor_2_msg(2, True))
    self.assertTrue(self.safety.get_volkswagen_brake_pressed_prev())

  def test_disengage_on_brake(self):
    self.safety.set_controls_allowed(1)
    self.safety.safety_rx_hook(self._motor_2_msg(2, True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_allow_brake_at_zero_speed(self):
    # Brake was already pressed
    self.safety.safety_rx_hook(self._motor_2_msg(0, True))
    self.safety.set_controls_allowed(1)

    self.safety.safety_rx_hook(self._motor_2_msg(2, True))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._motor_2_msg(2, False))  # reset no brakes

  def test_not_allow_brake_when_moving(self):
    # Brake was already pressed
    self.safety.safety_rx_hook(self._motor_2_msg(2, True))
    self.safety.safety_rx_hook(self._bremse_3_msg(100))
    self.safety.set_controls_allowed(1)

    self.safety.safety_rx_hook(self._motor_2_msg(2, True))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_disengage_on_gas(self):
    self.safety.safety_rx_hook(self._motor_1_msg(0))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._motor_1_msg(1))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_allow_engage_with_gas_pressed(self):
    self.safety.safety_rx_hook(self._motor_1_msg(1))
    self.safety.set_controls_allowed(True)
    self.safety.safety_rx_hook(self._motor_1_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.safety.safety_rx_hook(self._motor_1_msg(1))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_steer_safety_check(self):
    for enabled in [0, 1]:
      for t in range(-500, 500):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(t)
        if abs(t) > MAX_STEER or (not enabled and abs(t) > 0):
          self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(t)))
        else:
          self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(t)))

  def test_manually_enable_controls_allowed(self):
    test_manually_enable_controls_allowed(self)

  def test_spam_cancel_safety_check(self):
    BIT_CANCEL = 9
    BIT_SET = 16
    BIT_RESUME = 17
    self.safety.set_controls_allowed(0)
    self.assertTrue(self.safety.safety_tx_hook(self._gra_neu_msg(BIT_CANCEL)))
    self.assertFalse(self.safety.safety_tx_hook(self._gra_neu_msg(BIT_RESUME)))
    self.assertFalse(self.safety.safety_tx_hook(self._gra_neu_msg(BIT_SET)))
    # do not block resume if we are engaged already
    self.safety.set_controls_allowed(1)
    self.assertTrue(self.safety.safety_tx_hook(self._gra_neu_msg(BIT_RESUME)))

  def test_non_realtime_limit_up(self):
    self.safety.set_volkswagen_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(MAX_RATE_UP)))
    self._set_prev_torque(0)
    self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(-MAX_RATE_UP)))

    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(MAX_RATE_UP + 1)))
    self.safety.set_controls_allowed(True)
    self._set_prev_torque(0)
    self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(-MAX_RATE_UP - 1)))

  def test_non_realtime_limit_down(self):
    self.safety.set_volkswagen_torque_driver(0, 0)
    self.safety.set_controls_allowed(True)

  def test_against_torque_driver(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      for t in np.arange(0, DRIVER_TORQUE_ALLOWANCE + 1, 1):
        t *= -sign
        self.safety.set_volkswagen_torque_driver(t, t)
        self._set_prev_torque(MAX_STEER * sign)
        self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(MAX_STEER * sign)))

      self.safety.set_volkswagen_torque_driver(DRIVER_TORQUE_ALLOWANCE + 1, DRIVER_TORQUE_ALLOWANCE + 1)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(-MAX_STEER)))

    # spot check some individual cases
    for sign in [-1, 1]:
      driver_torque = (DRIVER_TORQUE_ALLOWANCE + 10) * sign
      torque_desired = (MAX_STEER - 10 * DRIVER_TORQUE_FACTOR) * sign
      delta = 1 * sign
      self._set_prev_torque(torque_desired)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(torque_desired)))
      self._set_prev_torque(torque_desired + delta)
      self.safety.set_volkswagen_torque_driver(-driver_torque, -driver_torque)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(torque_desired + delta)))

      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg((MAX_STEER - MAX_RATE_DOWN) * sign)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(0)))
      self._set_prev_torque(MAX_STEER * sign)
      self.safety.set_volkswagen_torque_driver(-MAX_STEER * sign, -MAX_STEER * sign)
      self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg((MAX_STEER - MAX_RATE_DOWN + 1) * sign)))

  def test_realtime_limits(self):
    self.safety.set_controls_allowed(True)

    for sign in [-1, 1]:
      self.safety.init_tests_volkswagen()
      self._set_prev_torque(0)
      self.safety.set_volkswagen_torque_driver(0, 0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(t)))
      self.assertFalse(self.safety.safety_tx_hook(self._hca_1_msg(sign * (MAX_RT_DELTA + 1))))

      self._set_prev_torque(0)
      for t in np.arange(0, MAX_RT_DELTA, 1):
        t *= sign
        self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(t)))

      # Increase timer to update rt_torque_last
      self.safety.set_timer(RT_INTERVAL + 1)
      self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(sign * (MAX_RT_DELTA - 1))))
      self.assertTrue(self.safety.safety_tx_hook(self._hca_1_msg(sign * (MAX_RT_DELTA + 1))))

  def test_torque_measurements(self):
    self.safety.safety_rx_hook(self._eps_1_msg(50))
    self.safety.safety_rx_hook(self._eps_1_msg(-50))
    self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.safety.safety_rx_hook(self._eps_1_msg(0))

    self.assertEqual(-50, self.safety.get_volkswagen_torque_driver_min())
    self.assertEqual(50, self.safety.get_volkswagen_torque_driver_max())

    self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_max())
    self.assertEqual(-50, self.safety.get_volkswagen_torque_driver_min())

    self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_max())
    self.assertEqual(0, self.safety.get_volkswagen_torque_driver_min())

  def test_rx_hook(self):
    # checksum checks
    # TODO: Would be ideal to check non-checksum non-counter messages as well,
    # but I'm not sure if we can easily validate Panda's simple temporal
    # reception-rate check here.
    for msg in [MSG_EPS_1]:
      self.safety.set_controls_allowed(1)
      if msg == MSG_EPS_1:
        to_push = self._eps_1_msg(0)
      self.assertTrue(self.safety.safety_rx_hook(to_push))
      to_push[0].RDHR ^= 0xFF
      self.assertFalse(self.safety.safety_rx_hook(to_push))
      self.assertFalse(self.safety.get_controls_allowed())

    # counter
    # reset wrong_counters to zero by sending valid messages
    for i in range(MAX_WRONG_COUNTERS + 1):
      self.cnt_eps_1 = 0
      if i < MAX_WRONG_COUNTERS:
        self.safety.set_controls_allowed(1)
        self.safety.safety_rx_hook(self._eps_1_msg(0))
      else:
        self.assertFalse(self.safety.safety_rx_hook(self._eps_1_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())

    # restore counters for future tests with a couple of good messages
    for i in range(2):
      self.safety.set_controls_allowed(1)
      self.safety.safety_rx_hook(self._eps_1_msg(0))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_fwd_hook(self):
    buss = list(range(0x0, 0x3))
    msgs = list(range(0x1, 0x800))
    blocked_msgs_0to2 = []
    blocked_msgs_2to0 = [MSG_HCA_1, MSG_LDW_1]
    for b in buss:
      for m in msgs:
        if b == 0:
          fwd_bus = -1 if m in blocked_msgs_0to2 else 2
        elif b == 1:
          fwd_bus = -1
        elif b == 2:
          fwd_bus = -1 if m in blocked_msgs_2to0 else 0

        # assume len 8
        self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, make_msg(b, m, 8)))


if __name__ == "__main__":
  unittest.main()
