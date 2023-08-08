import re
from datetime import datetime
import rospy
import glob
import os
from collections import defaultdict

def parse_file_content(file_content):
    useful_data = []
    timestamps = []

    # Regular expression patterns to match the desired information
    # good_expert_min_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| Good expert min (\d+\.\d+) All expert min (\d+\.\d+)"
    evaluation_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| \{'speed': '(\w+)', 'correction': \[(.*?)\], 'evaluation': \[(.*?)\]\}"
    speech_pattent = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| Robot says: (.*)"
    smile_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| Robot smiling at intensity (\d.\d) for duration"
    frown_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| Robot frowning at intensity (\d.\d) for duration"

    # good_expert_min_matches = re.findall(good_expert_min_pattern, file_content)
    evaluation_matches = re.findall(evaluation_pattern, file_content)
    speech_matches = re.findall(speech_pattent, file_content)
    smile_matches = re.findall(smile_pattern, file_content)
    frown_matches = re.findall(frown_pattern, file_content)

    # # Process the matches and extract the useful data with timestamps
    # for match in good_expert_min_matches:
    #     timestamp = match[0]
    #     timestamps.append(timestamp)
    #     useful_data.append(f"Good expert min {match[1]} All expert min {match[2]}")

    index = [match[1] for match in speech_matches].index("Almost done.")
    cutoff = speech_matches[index][0]

    for match in speech_matches[2:index]:
        timestamp = match[0]

        if timestamp >= cutoff:
            break
        
        speech = match[1]

        timestamps.append(timestamp)
        useful_data.append(speech)

    for match in evaluation_matches:
        timestamp = match[0]

        if timestamp >= cutoff:
            break

        timestamps.append(timestamp)
        speed = match[1]
        corrections = match[2].split(', ')
        evaluations = [int(eval_value) for eval_value in match[3].split(', ')]

        # speed evaluations
        results = []
        if speed == "slow":
            results.append(-1)
        elif speed == "good":
            results.append(0)
        elif speed == "fast":
            results.append(1)
        else:
            exit("Error")

        results.extend(evaluations)
        
        useful_data.append(results)

    for match in smile_matches:
        timestamp = match[0]

        if timestamp >= cutoff:
            break

        intensity = match[1]
        timestamps.append(timestamp)
        useful_data.append(intensity)

    for match in frown_matches:
        timestamp = match[0]

        if timestamp >= cutoff:
            break
        
        intensity = float(match[1])
        if intensity == 0:
            continue

        timestamps.append(timestamp)
        useful_data.append(-1 * intensity)
    
    processed_timestamps = [(parse_timestamp(timestamp)) for timestamp in timestamps]

    # Combine the normalized timestamps with the useful data
    data_with_timestamps = list(zip(processed_timestamps, useful_data))

    return data_with_timestamps

def parse_timestamp(timestamp):
    time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
    ros_time = round(rospy.Time.from_sec(time.timestamp()).to_sec() - offset, 3)
    return ros_time


log_dir = "/home/kevin/Documents/quori/src/openface2_ros/data/logs"
output_dir = "/home/kevin/Documents/quori/src/openface2_ros/processed_data"

offset = 1686000000

log_files = glob.glob(log_dir + "/*")
log_files.sort()

counter = defaultdict(int)
for log_file in log_files:
    # file_path = "/home/kevin/Documents/quori/src/openface2_ros/Participant_3_Round_3_Robot_3_Exercise_lateral_raises_Set_2.log"
    file_content = open(log_file, "r").read()

    data_with_timestamps = parse_file_content(file_content)

    file_name = log_file[log_file.rfind("/")+1:-4]
    pattern = r"Participant_(\d+)"
    match = re.search(pattern, file_name)

    output_subdir = output_dir + "/" + match[0] + "_" + str(counter[match[0]])
    counter[match[0]] += 1
    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    output_file  = open(output_subdir + "/log.txt", 'w')

    # Print the extracted useful data with timestamps
    for timestamp, data in data_with_timestamps:
        output_file.write(f"{timestamp} - {data}\n")

    output_file.close()