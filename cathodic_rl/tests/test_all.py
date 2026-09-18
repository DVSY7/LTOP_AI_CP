import _path_setup
from test_setting import test_setting
from test_reset import  test_reset
from test_step import test_step
from test_safety import test_voltage_safety_filter
from test_episode import test_episode_truncation
from test_env_checker import test_environment_format
from test_dqn import test_dqn_creation
from test_sac import test_sac_creation

import time


print(f"전체 테스트를 시작합니다.(3초)")

test_setting()
time.sleep(2)
test_reset()
time.sleep(2)
test_step()
time.sleep(2)
test_voltage_safety_filter()
time.sleep(2)
test_episode_truncation()
time.sleep(2)
test_environment_format()
time.sleep(2)
test_dqn_creation()
time.sleep(2)
test_sac_creation()
print(f"모든 테스트가 종료되었습니다.")



