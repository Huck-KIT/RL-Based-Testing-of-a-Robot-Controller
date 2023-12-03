import csv
import os
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

'''
MODES:
0 = get total runtime
1 = plot collisions
2 = plot high rewards eps
3 = plot heatmap of collisions 
4 = collision analysis laser fields
5 = collision analysis no laser fields
'''
MODE = 0
HIGH_REWARD = 100

TIMESTAMP = 0
EPISODE_NUMBER = 1
EPS_SINCE_LAST_INCIDENT = 2
WORKER_X_POS = 3
WORKER_Y_POS = 4
# WORKER_HEADING = 5
# WORKER_VISIBLE = 6
WORKER_VELO = 7
AGV_X_POS = 8
AGV_Y_POS = 9
AGV_HEADING = 10
AGV_VELO = 11
IS_EMERGENCY_STOP = 12
IS_COLLISION = 13
COLLISION_DIRECTION = 14  # left/right/front/rear
# DISTANCE = 15
REWARD = 16
# REWARD_PASSED_TO_RL = 17

LEFT = 0
RIGHT = 1
FRONT = 2
BACK = 3

KIT_RED = '#9F1D27'
KIT_RED_70 = '#BC6168'
KIT_RED_50 = '#CF8E93'
KIT_LILAC = '#9F0077'
KIT_LILAC_70 = '#BC4CA0'
KIT_LILAC_50 = '#CF7FBB'
KIT_ORANGE = '#DB9f1D'
KIT_ORANGE_70 = '#E6BC61'
KIT_ORANGE_50 = '#EDCF8E'
KIT_PALE_GREEN = '#81BD3B'
KIT_PALE_GREEN_70 = '#A7D176'
KIT_PALE_GREEN_50 = '#C0DE9D'
KIT_CYAN = '#4FA9E5'
KIT_CYAN_70 = '#84C3ED'
KIT_CYAN_50 = '#A7D4F2'
KIT_BLUE = '#4563A9'
KIT_BLUE_70 = '#7D92C3'
KIT_BLUE_50 = '#A2B1D4'


def get_runtime(filename):
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    date_format_backup = '%Y-%m-%d %H:%M:%S'
    with open(filename, 'r') as report:
        data = csv.reader(report)
        next(data)
        start = 0
        for line in data:
            if start == 0:
                try:
                    start = datetime.datetime.strptime(line[TIMESTAMP], date_format)
                except ValueError:
                    start = datetime.datetime.strptime(line[TIMESTAMP], date_format_backup)
            if int(line[EPISODE_NUMBER]) > 20000:
                break
            else:
                try:
                    end = datetime.datetime.strptime(line[TIMESTAMP], date_format)
                except ValueError:
                    end = datetime.datetime.strptime(line[TIMESTAMP], date_format_backup)
    return end - start


def get_heatmap():
    x_values_worker = []
    y_values_worker = []
    x_values_agv = []
    y_values_agv = []
    for i in range(3):
        root = Tk()
        root.withdraw()
        filename = askopenfilename()
        root.update()
        root.destroy()

        with open(filename, 'r') as report:
            data = csv.reader(report)
            next(data)  # skip title-row
            for line in data:
                if int(line[EPISODE_NUMBER]) > 20000:
                    break
                if int(line[IS_COLLISION]) == 1:
                    x_values_worker.append(float(line[WORKER_X_POS]))
                    y_values_worker.append(float(line[WORKER_Y_POS]))
                    x_values_agv.append(float(line[AGV_X_POS]))
                    y_values_agv.append(float(line[AGV_Y_POS]))
    return x_values_worker, y_values_worker, x_values_agv, y_values_agv


def get_collisions(filename):
    eps_with_collision = []
    time_between = []
    collisions = []
    last_collision = 0
    with open(filename, 'r') as report:
        data = csv.reader(report)
        next(data)  # skip title-row
        current_ep_num = 0
        ep_had_collision = False
        sum_collisions = 0
        for line in data:
            if int(line[EPISODE_NUMBER]) > 20000:
                break
            if line[EPISODE_NUMBER] != current_ep_num:
                # found a new logged episode, reset the marker
                ep_had_collision = False
                current_ep_num = line[EPISODE_NUMBER]
            if (int(line[IS_COLLISION]) == 1) and (not ep_had_collision):
                # collision found in ep and did not log that collision yet
                eps_with_collision.append(int(line[EPISODE_NUMBER]))
                sum_collisions += 1
                collisions.append(sum_collisions)
                time_between.append(int(line[EPISODE_NUMBER])-last_collision)
                last_collision = int(line[EPISODE_NUMBER])
                ep_had_collision = True
                print(line[EPISODE_NUMBER], line[REWARD])
    return eps_with_collision, time_between, collisions


def get_high_reward_eps(filename):
    eps_with_high_reward = []
    time_between = []
    high_reward_eps = []
    with open(filename, 'r') as report:
        data = csv.reader(report)
        next(data)  # skip title-row
        current_ep_num = 0
        ep_had_high_reward = False
        total_high_reward_episodes = 0
        for line in data:
            if int(line[EPISODE_NUMBER]) > 20000:
                break
            if line[EPISODE_NUMBER] != current_ep_num:
                # found a new logged episode, reset the marker
                ep_had_high_reward = False
                current_ep_num = line[EPISODE_NUMBER]
            if (float(line[REWARD]) >= HIGH_REWARD) and (not ep_had_high_reward):
                # high reward found in ep and did not log that yet
                eps_with_high_reward.append(int(line[EPISODE_NUMBER]))
                total_high_reward_episodes += 1
                high_reward_eps.append(total_high_reward_episodes)
                time_between.append(int(line[EPS_SINCE_LAST_INCIDENT]))
                ep_had_high_reward = True
    return eps_with_high_reward, time_between, high_reward_eps


def analyse_estop_collision(ep_data, lf_type):
    agv_length = 0.32

    if lf_type == 1:
        rlf = 1.25 / 2  # radius laser field
    elif lf_type == 2:
        rlf = 2 / 2
    else:
        return False

    for i in range(len(ep_data) - 1):
        if (int(ep_data[-i][IS_EMERGENCY_STOP]) == 1) and (int(ep_data[-i - 1][IS_EMERGENCY_STOP]) == 0):
            # AGV
            angle_new = float(ep_data[-i][AGV_HEADING])
            point1_x_new = np.cos(angle_new) * agv_length - np.sin(angle_new) * rlf + float(ep_data[-i][AGV_X_POS])
            point1_y_new = np.sin(angle_new) * agv_length + np.cos(angle_new) * rlf + float(ep_data[-i][AGV_X_POS])
            point2_x_new = np.cos(angle_new) * agv_length - np.sin(angle_new) * (-rlf) + float(ep_data[-i][AGV_X_POS])
            point2_y_new = np.sin(angle_new) * agv_length + np.cos(angle_new) * (-rlf) + float(ep_data[-i][AGV_X_POS])

            dx_new = point1_x_new - point2_x_new
            dy_new = point1_y_new - point2_y_new
            m_new = dy_new / dx_new
            b_new = point2_y_new - m_new * point2_x_new

            angle_old = float(ep_data[-i - 1][AGV_HEADING])
            point1_x_old = np.cos(angle_old) * agv_length - np.sin(angle_old) * rlf + float(ep_data[-i - 1][AGV_X_POS])
            point1_y_old = np.sin(angle_old) * agv_length + np.cos(angle_old) * rlf + float(ep_data[-i - 1][AGV_X_POS])
            point2_x_old = np.cos(angle_old) * agv_length - np.sin(angle_old) * (-rlf) + float(
                ep_data[-i - 1][AGV_X_POS])
            point2_y_old = np.sin(angle_old) * agv_length + np.cos(angle_old) * (-rlf) + float(
                ep_data[-i - 1][AGV_X_POS])

            dx_old = point1_x_old - point2_x_old
            dy_old = point1_y_old - point2_y_old
            m_old = dy_old / dx_old
            b_old = point2_y_old - m_old * point2_x_old

            # WORKER
            # first point inside the laser field
            point_inside_x = float(ep_data[-i][WORKER_X_POS])
            point_inside_y = float(ep_data[-i][WORKER_Y_POS])
            # last point outside the laser field
            point_outside_x = float(ep_data[-i - 1][WORKER_X_POS])
            point_outside_y = float(ep_data[-i - 1][WORKER_Y_POS])
            # y = mx + b
            dx_worker = point_inside_x - point_outside_x
            if dx_worker == 0:
                dx_worker = 0.000001
            dy_worker = point_inside_y - point_outside_y
            m_worker = dy_worker / dx_worker
            b_worker = point_outside_y - m_worker * point_outside_x

            # INTERCEPTS
            if np.abs(m_worker - m_new) > 0.01:
                intercept_new_x = (b_worker - b_new) / (m_new - m_worker)
                intercept_new_y = m_new * intercept_new_x + b_new
                if ((min(point1_x_new, point2_x_new) <= intercept_new_x <= max(point1_x_new, point2_x_new)) and
                        (min(point1_y_new, point2_y_new) <= intercept_new_y <= max(point1_y_new, point2_y_new))):
                    return True

            if np.abs(m_worker - m_old) > 0.01:
                intercept_old_x = (b_worker - b_old) / (m_old - m_worker)
                intercept_old_y = m_old * intercept_old_x + b_old
                if ((min(point1_x_old, point2_x_old) <= intercept_old_x <= max(point1_x_old, point2_x_old)) and
                        (min(point1_y_old, point2_y_old) <= intercept_old_y <= max(point1_y_old, point2_y_old))):
                    return True

            return False


def analyse_collisions():
    scenario_type = ""

    # ##################################################################
    # ######################## CASE DESCRIPTION ########################
    # ##################################################################

    case1_right = 0  # seitliches Einlaufen ohne Auslösen der Laserfelder von rechts
    case1_left = 0  # seitliches Einlaufen ohne Auslösen der Laserfelder von links
    case2a_right = 0  # Auslösen Laserfeld, verlassen Laserfeld, Wiederanlauf AGV, Kontakt von rechts
    case2a_left = 0  # Auslösen Laserfeld, verlassen Laserfeld, Wiederanlauf AGV, Kontakt von links
    case2b_right = 0  # Auslösen Laserfeld, verlassen Laserfeld, kein Wiederanlauf AGV, Kontakt von rechts
    case2b_left = 0  # Auslösen Laserfeld, verlassen Laserfeld, kein Wiederanlauf AGV, Kontakt von links
    case3a = 0  # Auslösen Laserfeld, min Geschwindigkeit Werker erreicht
    case3b = 0  # Auslösen Laserfeld, min Geschwindigkeit Werker nicht erreicht
    case4a = 0  # Einlauf in das Laserfeld über die gerade Kante, min Geschwindigkeit Werker erreicht
    case4b = 0  # Einlauf in das Laserfeld über die gerade Kante, min Geschwindigkeit Werker nicht erreicht

    case1_right_first = 0
    case1_left_first = 0
    case2a_right_first = 0
    case2a_left_first = 0
    case2b_right_first = 0
    case2b_left_first = 0
    case3a_first = 0
    case3b_first = 0
    case4a_first = 0
    case4b_first = 0
    case_other = 0

    case1_list = []

    count_collisions = 0
    case2_steps_since_estop = []
    case3_worker_velo = []
    case3_emergency_steps = []

    for report_count in range(3):
        root = Tk()
        root.withdraw()
        filename = askopenfilename()
        root.update()
        root.destroy()
        if report_count == 0:
            if "1m" in filename:
                lf_type = 1
                scenario_type = "0.625m Laserfeldradius"
            elif "2m" in filename:
                lf_type = 2
                scenario_type = "1m Laserfeldradius"
            else:
                lf_type = 0
                scenario_type = "Keine Laserfelder"
            if "RS" in filename:
                scenario_type += ", Random Sampling"
            elif "max" in filename:
                scenario_type += ", RL (max reward)"
            elif "acc" in filename:
                scenario_type += ", RL (acc reward)"
#            if "v2" in filename:
#                scenario_type += ", v2"

        with open(filename, 'r') as report:
            data = csv.reader(report)
            next(data)  # skip title-row

            current_ep_data = []  # store to the rows of the current episode here to trace back the lines later
            current_ep_num = 0
            estop_steps = 0
            steps_without_estop = 0

            for index, line in enumerate(data):

                current_ep_data.append(line)

                if int(line[EPISODE_NUMBER]) > 20000:
                    break

                if line[EPISODE_NUMBER] != current_ep_num:
                    # reached next episode
                    current_ep_num = line[EPISODE_NUMBER]
                    current_ep_data = []
                    estop_steps = 0
                    steps_without_estop = 0
                    ep_had_collision = False
                    ep_had_prev_estop = False

                if int(line[IS_EMERGENCY_STOP]) == 0:
                    estop_steps = 0
                    steps_without_estop += 1

                if ep_had_collision or (int(line[IS_COLLISION]) == 0):
                    # episode's collision found already or had none yet, nothing to do (yet)
                    if int(line[IS_EMERGENCY_STOP]) == 1:
                        ep_had_prev_estop = True
                        estop_steps += 1
                        steps_without_estop = 0
                    continue

                # line is reached if collision found in ep and did not log that collision yet
                ep_had_collision = True
                count_collisions += 1

                if int(line[IS_EMERGENCY_STOP]) == 1:
                    # collision during emergency stop situation
                    if int(current_ep_data[-2][IS_EMERGENCY_STOP]) == 1:
                        # laser field triggered at least 1 step beforehand
                        case3_worker_velo.append(float(line[WORKER_VELO]))
                        case3_emergency_steps.append(estop_steps)
                        if analyse_estop_collision(current_ep_data, lf_type):
                            if float(line[WORKER_VELO]) > 0.01:
                                case4a += 1
                                if int(line[EPISODE_NUMBER]) < case4a_first or case4a_first == 0:
                                    case4a_first = int(line[EPISODE_NUMBER])
                            else:
                                case4b += 1
                                if int(line[EPISODE_NUMBER]) < case4b_first or case4b_first == 0:
                                    case4b_first = int(line[EPISODE_NUMBER])
                        else:
                            if float(line[WORKER_VELO]) > 0.01:
                                case3a += 1
                                if int(line[EPISODE_NUMBER]) < case3a_first or case3a_first == 0:
                                    case3a_first = int(line[EPISODE_NUMBER])
                            else:
                                case3b += 1
                                if int(line[EPISODE_NUMBER]) < case3b_first or case3b_first == 0:
                                    case3b_first = int(line[EPISODE_NUMBER])
                        continue

                # not a collision during emergency stop situation
                for i in range(1, 5):
                    # check the current and the previous 3 eps if a side collision was detected
                    if current_ep_data[-i][COLLISION_DIRECTION].strip()[RIGHT] == '1':
                        if not ep_had_prev_estop:
                            # there was no emergency stop beforehand
                            case1_right += 1
                            case1_list.append(int(line[EPISODE_NUMBER]))
                            if int(line[EPISODE_NUMBER]) < case1_right_first or case1_right_first == 0:
                                case1_right_first = int(line[EPISODE_NUMBER])
                        else:
                            # there was an emergency stop beforehand
                            case2_steps_since_estop.append(steps_without_estop)
                            j = 1
                            while True:
                                if int(current_ep_data[-i - j][IS_EMERGENCY_STOP]) == 1:
                                    case2b_right += 1
                                    if int(line[EPISODE_NUMBER]) < case2b_right_first or case2b_right_first == 0:
                                        case2b_right_first = int(line[EPISODE_NUMBER])
                                    break
                                # trace back till the emergency stop occurred:
                                if float(current_ep_data[i - j][AGV_VELO]) == 0:
                                    case2a_right += 1
                                    if int(line[EPISODE_NUMBER]) < case2a_right_first or case2a_right_first == 0:
                                        case2a_right_first = int(line[EPISODE_NUMBER])
                                    break
                                j += 1
                        break
                    elif current_ep_data[-i][COLLISION_DIRECTION].strip()[LEFT] == '1':
                        if not ep_had_prev_estop:
                            # there was no emergency stop beforehand
                            case1_left += 1
                            if int(line[EPISODE_NUMBER]) < case1_left_first or case1_left_first == 0:
                                case1_left_first = int(line[EPISODE_NUMBER])
                        else:
                            print(line[EPISODE_NUMBER])
                            # there was an emergency stop beforehand
                            case2_steps_since_estop.append(steps_without_estop)
                            j = 1
                            while True:
                                if int(current_ep_data[-i - j][IS_EMERGENCY_STOP]) == 1:
                                    case2b_left += 1
                                    if int(line[EPISODE_NUMBER]) < case2b_left_first or case2b_left_first == 0:
                                        case2b_left_first = int(line[EPISODE_NUMBER])
                                    break
                                # trace back till the emergency stop occurred:
                                if float(current_ep_data[i - j][AGV_VELO]) == 0:
                                    case2a_left += 1
                                    if int(line[EPISODE_NUMBER]) < case2a_left_first or case2a_left_first == 0:
                                        case2a_left_first = int(line[EPISODE_NUMBER])
                                    break
                                j += 1
                        break
                    elif i == 4:
                        case_other += 1

    plt.figure(figsize=(10, 10))
    plt.hist(case3_worker_velo, bins=30)
    plt.title("Worker velocity at collision (front collision, " + scenario_type + ")")
    plt.xlabel("Worker velocity")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(case2_steps_since_estop, bins=60)
    plt.title("Steps since emergency stop (side collision, " + scenario_type + ")")
    plt.xlabel("Steps since emergency stop")
    plt.ylabel("Frequency")
    plt.show()

    steps, counts = np.unique(case3_emergency_steps, return_counts=True)
    plt.figure(figsize=(10, 10))
    plt.bar(steps, counts, align='center')
    if len(steps) > 0:
        plt.xticks(range(0, np.max(steps) + 1))
    plt.title("Emerg. stop steps before front collision (" + scenario_type + ")")
    plt.xlabel("Emerg. stop steps taken")
    plt.ylabel("Frequency")
    plt.show()

    collision_types = [np.ceil(case1_right / 3), np.ceil(case1_left / 3), np.ceil(case2a_right / 3),
                       np.ceil(case2a_left / 3), np.ceil(case2b_right / 3), np.ceil(case2b_left / 3),
                       np.ceil(case4a / 3), np.ceil(case4b / 3), np.ceil(case3a / 3), np.ceil(case3b / 3),
                       np.ceil(case_other / 3), np.ceil(count_collisions / 3)]
    plt.figure(figsize=(10, 10))

    graph = plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'N/C',
                     'total'],
                    collision_types, align='center',
                    color=[KIT_PALE_GREEN, KIT_PALE_GREEN_70, KIT_BLUE, KIT_BLUE_50, KIT_CYAN, KIT_CYAN_70,
                           KIT_ORANGE, KIT_ORANGE_50, KIT_LILAC, KIT_LILAC_50, 'tab:gray', KIT_RED],
                    zorder=2)

    plt.title("Identified collisions by classes (" + scenario_type + ")", fontsize=17)
    plt.xlabel("Collision classes", fontsize=17)
    plt.ylabel("Number of collisions", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylim([0, 6050])
    plt.grid(axis='y', color='gray', linestyle='dashed', zorder=1)

    j = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height + 5,
                 int(collision_types[j]),
                 ha='center',
                 fontsize=15)
        j += 1

    plt.show()

    print("case1_right:", case1_right)
    print("case1_left:", case1_left)
    print("case2a_right:", case2a_right)
    print("case2a_left:", case2a_left)
    print("case2b_right:", case2b_right)
    print("case2b_left:", case2b_left)
    print("case3a:", case3a)
    print("case3b:", case3b)
    print("case4a:", case4a)
    print("case4b:", case4b)
    print("total collisions:", count_collisions)
    print("other: ", case_other, " (", round(100/count_collisions*case_other, 2), "%)", sep='')
    print(case1_list)


def analyse_collisions_none():
    scenario_type = ""

    # ##################################################################
    # ######################## CASE DESCRIPTION ########################
    # ##################################################################

    case_front = 0
    case_front_right = 0
    case_right = 0
    case_back_right = 0
    case_back = 0
    case_back_left = 0
    case_left = 0
    case_front_left = 0
    case_other = 0

    case_front_first = 0
    case_front_right_first = 0
    case_right_first = 0
    case_back_right_first = 0
    case_back_first = 0
    case_back_left_first = 0
    case_left_first = 0
    case_front_left_first = 0

    count_collisions = 0
    worker_velo = []

    for report_count in range(3):
        root = Tk()
        root.withdraw()
        filename = askopenfilename()
        root.update()
        root.destroy()
        if report_count == 0:
            if "1m" in filename:
                lf_type = 1
                scenario_type = "0.625m laser field radius"
            elif "2m" in filename:
                lf_type = 2
                scenario_type = "1m laser field radius"
            else:
                lf_type = 0
                scenario_type = "keine Laserfelder"
            if "bl" in filename:
                scenario_type += ", Random Sampling"
            else:
                scenario_type += ", Reinforcement Learning"
#            if "v2" in filename:
#                scenario_type += ", v2"

        with open(filename, 'r') as report:
            data = csv.reader(report)
            next(data)  # skip title-row

            current_ep_data = []  # store to the rows of the current episode here to trace back the lines later
            current_ep_num = 0

            for index, line in enumerate(data):

                current_ep_data.append(line)

                if int(line[EPISODE_NUMBER]) > 20000:
                    break

                if line[EPISODE_NUMBER] != current_ep_num:
                    # reached next episode
                    current_ep_num = line[EPISODE_NUMBER]
                    current_ep_data = []
                    ep_had_collision = False

                if ep_had_collision or (int(line[IS_COLLISION]) == 0):
                    # episode's collision found already or had none yet, nothing to do (yet)
                    continue

                # line is reached if collision found in ep and did not log that collision yet
                ep_had_collision = True
                count_collisions += 1
                worker_velo.append(float(line[WORKER_VELO]))

                for i in range(1, 5):
                    # check the current and the previous 3 eps if a side collision was detected
                    if current_ep_data[-i][COLLISION_DIRECTION].strip()[FRONT] == '1':
                        if current_ep_data[-i][COLLISION_DIRECTION].strip()[RIGHT] == '1':
                            case_front_right += 1
                            if int(line[EPISODE_NUMBER]) < case_front_right_first or case_front_right_first == 0:
                                case_front_right_first = int(line[EPISODE_NUMBER])
                        elif current_ep_data[-i][COLLISION_DIRECTION].strip()[LEFT] == '1':
                            case_front_left += 1
                            if int(line[EPISODE_NUMBER]) < case_front_left_first or case_front_left_first == 0:
                                case_front_left_first = int(line[EPISODE_NUMBER])
                        else:
                            case_front += 1
                            if int(line[EPISODE_NUMBER]) < case_front_first or case_front_first == 0:
                                case_front_first = int(line[EPISODE_NUMBER])
                        break
                    elif current_ep_data[-i][COLLISION_DIRECTION].strip()[BACK] == '1':
                        if current_ep_data[-i][COLLISION_DIRECTION].strip()[RIGHT] == '1':
                            case_back_right += 1
                            if int(line[EPISODE_NUMBER]) < case_back_right_first or case_back_right_first == 0:
                                case_back_right_first = int(line[EPISODE_NUMBER])
                        elif current_ep_data[-i][COLLISION_DIRECTION].strip()[LEFT] == '1':
                            case_back_left += 1
                            if int(line[EPISODE_NUMBER]) < case_back_left_first or case_back_left_first == 0:
                                case_back_left_first = int(line[EPISODE_NUMBER])
                        else:
                            case_back += 1
                            if int(line[EPISODE_NUMBER]) < case_back_first or case_back_first == 0:
                                case_back_first = int(line[EPISODE_NUMBER])
                        break
                    if current_ep_data[-i][COLLISION_DIRECTION].strip()[RIGHT] == '1':
                        case_right += 1
                        if int(line[EPISODE_NUMBER]) < case_right_first or case_right_first == 0:
                            case_right_first = int(line[EPISODE_NUMBER])
                        break
                    if current_ep_data[-i][COLLISION_DIRECTION].strip()[LEFT] == '1':
                        case_left += 1
                        if int(line[EPISODE_NUMBER]) < case_left_first or case_left_first == 0:
                            case_left_first = int(line[EPISODE_NUMBER])
                        break
                    elif i == 4:
                        case_other += 1

    plt.figure(figsize=(10, 10))
    plt.hist(worker_velo, bins=30)
    plt.title("Worker's velocity in the moment of a collision")
    plt.xlabel("Worker's velocity")
    plt.ylabel("Frequency")
    plt.show()

    collision_types = [case_front, case_back, case_front_right, case_front_left, case_right, case_left,
                       case_back_right, case_back_left, case_other, count_collisions]
    plt.figure(figsize=(10, 10))
    graph = plt.bar(['front\n\n\n' + str(case_front_first), 'back\n\n\n' + str(case_back_first),
                     'front\n right\n\n' + str(case_front_right_first),
                     'front\n left\n\n' + str(case_front_left_first),
                     'right\n\n\n' + str(case_right_first),
                     'left\n\n\n' + str(case_left_first),
                     'back\n right\n\n' + str(case_back_right_first),
                     'back\n left\n\n' + str(case_back_left_first),
                     'other\n\n\n' + str(case_other),
                     'total\n\n\n ← first\n occurrence'],
                    collision_types, align='center',
                    color=[KIT_PALE_GREEN, KIT_PALE_GREEN_70, KIT_BLUE, KIT_BLUE_50, KIT_ORANGE, KIT_ORANGE_50,
                           KIT_CYAN, KIT_CYAN_70, 'tab:gray', KIT_RED],
                    zorder=2)
    plt.title("Klassifikation der Kollisionen (" + scenario_type + ")", fontsize=17)
    plt.ylabel("Häufigkeit", fontsize=17)
    # plt.ylim([0, 100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', color='gray', linestyle='dashed', zorder=1)

    j = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height + 3,
                 collision_types[j],
                 ha='center',
                 fontsize=15)
        j += 1

    plt.show()

    print("other: ", case_other, " (", round(100 / count_collisions * case_other, 2), "%)", sep='')


def calc_median(data, window=0):
    medians = []
    for i in range(len(data)):
        if window == 0:
            medians.append(np.median(np.array(data[0:i])))
        else:
            medians.append(np.median(np.array(data[max(0, i - window):i])))
    return medians


def calc_mean(data, window=0):
    means = []
    for i in range(len(data)):
        if window == 0:
            means.append(np.mean(np.array(data[0:i])))
        else:
            means.append(np.mean(np.array(data[max(0, i - window):i])))
    return means


def calc_std_dev(data, window):
    std = []
    for i in range(len(data)):
        if window == 0:
            std.append(np.std(np.array(data[0:i])))
        else:
            std.append(np.std(np.array(data[max(0, i - window):i])))
    return std


def plot_heatmap(x_worker, y_worker, x_agv, y_agv, mode='agv', scenario='crossing'):
    color = 'tab:red'
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.set_xlabel('x-coordinate')
    ax1.set_ylabel('y-coordinate')

    if mode == 'worker':
        ax1.scatter(x_worker, y_worker, s=4, color=color, label='Locations of collisions')
    else:
        ax1.scatter(x_agv, y_agv, s=4, color=color, label='Locations of collisions')

    if scenario == 'overtaking':
        ax1.scatter(x=[12.575], y=[0.4], s=100, c='g', label='Start position AGV')
        ax1.scatter(x=[7.95], y=[0.38], s=100, c='b', label='Start position worker')
        ax1.add_patch(Rectangle((-0.7, -1.605), 16, 0.15, alpha=0.5, facecolor='tab:gray', label='Walls'))
        ax1.add_patch(Rectangle((-0.7, 2.255), 16, 0.15, alpha=0.5, facecolor='tab:gray'))
        ax1.set_xlim([17.3, -2.7])
        ax1.set_ylim([2.875, -2.125])
    elif scenario == 'head-on':
        ax1.scatter(x=[12.575], y=[0.4], s=100, c='g', label='Start position AGV')
        ax1.scatter(x=[5.95], y=[0.38], s=100, c='b', label='Start position worker')
        ax1.add_patch(Rectangle((-0.7, -1.605), 16, 0.15, alpha=0.5, facecolor='tab:gray', label='Walls'))
        ax1.add_patch(Rectangle((-0.7, 2.255), 16, 0.15, alpha=0.5, facecolor='tab:gray'))
        ax1.set_xlim([17.3, -2.7])
        ax1.set_ylim([2.875, -2.125])
    else:
        ax1.scatter(x=[0.45], y=[-5], s=100, c='g', label='Start position AGV')
        ax1.scatter(x=[3.9254], y=[-1.05], s=100, c='b', label='Start position worker')
        ax1.add_patch(Rectangle((1.625, -2.1), 1.2, 0.3, alpha=0.5, facecolor='tab:gray', label='Rack'))
        ax1.set_xlim([7.0, -2.5])
        ax1.set_ylim([2.5, -7.0])

    plt.rcParams['figure.figsize'] = [1, 1]
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
    ax1.set_aspect(1)
    plt.show()


def plot_all_collisions(data):
    if len(data) == 2:
        plt.plot(data[0], data[1])
        plt.show()

    if len(data) == 3:
        color = 'tab:blue'
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Episode", fontsize=17)
        ax1.set_ylabel('Episodes between collisions', color=color, fontsize=16)
        ax1.plot(data[0], data[1], color=color)
        ax1.set_xlim([0, 20000])
        #ax1.set_ylim([0, 1000])
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)
        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel("Number of collisions (accumulated)", color=color, fontsize=16)
        ax2.plot(data[0], data[2], color=color)
        #ax2.set_ylim([0, 100])
        ax2.tick_params(axis='y', labelcolor=color)
        plt.xticks(fontsize=15)

        fig.tight_layout()
        plt.show()


def plot_high_rewards(data):
    if len(data) == 2:
        plt.plot(data[0], data[1])
        plt.show()

    if len(data) == 3:
        color = 'tab:red'
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("episode#")
        ax1.set_ylabel('eps between two high reward eps', color=color)
        ax1.plot(data[0], data[1], color=color)
        ax1.set_xlim([0, 20000])
        #ax1.set_ylim([0, 500])
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel("#eps with reward greater than " + str(HIGH_REWARD), color=color)
        ax2.plot(data[0], data[2], color=color)
        #ax2.set_ylim([0, 10000])
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()


def plot_med_mean_sdev(ep_numbers, data, plot_median=True, plot_mean=False, plot_std_dev=False, window=0):
    y_max = 1000

    color = 'tab:blue'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("episode number")
    ax1.set_ylabel('episodes between two collisions', color=color)
    ax1.plot(ep_numbers, data, color=color)
    #ax1.set_xlim([0, 20000])
    ax1.set_ylim([0, y_max])
    ax1.tick_params(axis='y', labelcolor=color)

    if plot_median:
        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel("median", color=color, loc='bottom')
        ax2.plot(ep_numbers, calc_median(data, window), color=color)
        #ax2.set_ylim([0, y_max])
        if plot_mean or plot_std_dev:
            ax2.tick_params(axis='y', labelcolor='tab:gray')
        else:
            ax2.tick_params(axis='y', labelcolor=color)
    if plot_mean:
        ax3 = ax1.twinx()

        color = 'tab:orange'
        ax3.set_ylabel("mean", color=color, loc='center')
        ax3.plot(ep_numbers, calc_mean(data, window), color=color)
        #ax3.set_ylim([0, y_max])
        if plot_median or plot_std_dev:
            ax3.tick_params(axis='y', labelcolor='tab:gray')
        else:
            ax3.tick_params(axis='y', labelcolor=color)

    if plot_std_dev:
        ax4 = ax1.twinx()

        color = 'tab:purple'
        ax4.set_ylabel("standard deviation", color=color, loc='top')
        ax4.plot(ep_numbers, calc_std_dev(data, window), color=color)
        #ax4.set_ylim([0, y_max])
        if plot_median or plot_mean:
            ax4.tick_params(axis='y', labelcolor='tab:gray')
        else:
            ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if 0 <= MODE <= 3:
        # file dialog
        root = Tk()
        root.withdraw()
        reportFileName = askopenfilename()
        root.update()
        root.destroy()

    if MODE == 0:
        print(get_runtime(reportFileName))
    elif MODE == 1:
        ep_numbers, time_between_eps, total_collisions = get_collisions(reportFileName)
        print(ep_numbers)
        print(len(ep_numbers))

        # plot all collisions
        plot_all_collisions([ep_numbers, time_between_eps, total_collisions])

        # plot median, mean, standard deviation
        plot_med_mean_sdev(ep_numbers, time_between_eps, True, True, True)
    elif MODE == 2:
        ep_numbers, time_between_eps, total_high_reward_eps = get_high_reward_eps(reportFileName)
        print(ep_numbers)

        # plot high reward episodes
        plot_high_rewards([ep_numbers, time_between_eps, total_high_reward_eps])

        # plot median, mean, standard deviation
        plot_med_mean_sdev(ep_numbers, time_between_eps, True, True, True)
    elif MODE == 3:
        if "crossing" in os.path.basename(reportFileName):
            plot_heatmap(*get_heatmap(reportFileName), mode='agv', scenario='crossing')
        elif "overtaking" in os.path.basename(reportFileName):
            plot_heatmap(*get_heatmap(reportFileName), mode='worker', scenario='overtaking')
        elif "head-on" in os.path.basename(reportFileName) or "headon" in os.path.basename(reportFileName):
            plot_heatmap(*get_heatmap(reportFileName), mode='agv', scenario='head-on')
        else:
            plot_heatmap(*get_heatmap(reportFileName))
    elif MODE == 4:
        analyse_collisions()
    elif MODE == 5:
        analyse_collisions_none()
    else:
        print("MODE does not exist!")
