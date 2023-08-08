import glob
import csv
import re
from datetime import datetime
import rospy
import matplotlib.pyplot as plt

def process_rpe_csv(rpe_csv):
    rpes = dict()

    with open(rpe_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            row[1] = int(row[1])

            if row[1] not in rpes:
                rpes[row[1]] = dict()
            if row[2] not in rpes[row[1]]:
                rpes[row[1]][row[2]] = dict()

            rpes[row[1]][row[2]]["bicep_curls"] = [int(row[3]), int(row[4])]
            rpes[row[1]][row[2]]["lateral_raises"] = [int(row[5]), int(row[6])] 
    
    return rpes

def extract_timestamps(file_path):

    file_content = open(file_path, "r").read()

    # Define the pattern to match the log entries
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| Robot says: (Start lateral raises now|Start bicep curls now|Rest\.)"

    # Find all matches using regex
    matches = re.findall(pattern, file_content)[:2]

    # Extract the timestamps from the matches
    dates = [match[0] for match in matches]

    return dates

def convert_rostime(timestamps):
    start_date_time_obj = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S,%f')
    end_date_time_obj = datetime.strptime(timestamps[1], '%Y-%m-%d %H:%M:%S,%f')

    start_ros_time = rospy.Time.from_sec(start_date_time_obj.timestamp()).to_sec() - offset
    end_ros_time = rospy.Time.from_sec(end_date_time_obj.timestamp()).to_sec() - offset

    return [round(start_ros_time, 3), round(end_ros_time, 3)]

def process_log(log_file, all_timestamps):
    print("processing " + log_file)

    # Extracting participant ID
    participant_id = int(re.search(r'Participant_(\d+)', log_file).group(1))

    # Extracting round number
    round_number = re.search(r'Round_(\d+)', log_file).group(1)

    # Extracting exercise name
    exercise_name = re.search(r'Exercise_(.+)_Set', log_file).group(1)

    # Extracting set number
    set_number = int(re.search(r'Set_(\d+)', log_file).group(1))

    # Extract timestamps
    timestamps = extract_timestamps(log_file)
    timestamps = convert_rostime(timestamps)

    if participant_id not in all_timestamps:
        all_timestamps[participant_id] = dict()
    if round_number not in all_timestamps[participant_id]:
        all_timestamps[participant_id][round_number] = dict()
    if exercise_name not in all_timestamps[participant_id][round_number]:
        all_timestamps[participant_id][round_number][exercise_name] = dict()

    all_timestamps[participant_id][round_number][exercise_name][set_number] = timestamps

def get_rpes(all_timestamps, rpes, output_dir):
    fig, axes = plt.subplots(len(all_timestamps), 1, figsize=(8, 5*len(all_timestamps),))


    for participant_id in all_timestamps:

        print("id", participant_id)

        output_file = output_dir + "/Participant_" + str(participant_id) + ".txt"

        xs = []
        ys = []  

        for round_number in all_timestamps[participant_id]:
            for exercise in ["bicep_curls", "lateral_raises"]:
                timestamps1 = all_timestamps[participant_id][round_number][exercise][1]
                timestamps2 = all_timestamps[participant_id][round_number][exercise][2]

                assert(timestamps1[0] < timestamps1[1])
                assert(timestamps1[1] < timestamps2[0])
                assert(timestamps2[0] < timestamps2[1])

                rpe1 = rpes[participant_id][round_number][exercise][0]
                rpe2 = rpes[participant_id][round_number][exercise][1]

                xs.extend(timestamps1)
                ys.extend([rpe1, rpe1])

                xs.extend(timestamps2)
                ys.extend([rpe2, rpe2])

        data = list(zip(xs, ys))
        sorted_data = sorted(data, key=lambda x: x[0])
        xs, ys = zip(*sorted_data)

        # set the initial point of rpe to be 1
        ys = list(ys)
        ys[0] = 1

        # add the resting effects
        for i in range(2, len(ys), 2):
            # prev_point = xs[i-1], ys[i-1]
            # curr_point = xs[i], ys[i]

            time_diff = xs[i] - xs[i-1]

            diff = time_diff * 0.041

            ys[i] = ys[i-1] - diff

        initial = xs[0]
        xs = [x-initial for x in xs]

        with open(output_file, 'w') as file:
            file.write(str(xs).replace('(', '').replace(')', '') + '\n')
            file.write(str(ys).replace('[', '').replace(']', '') + '\n')


        index = participant_id-8
        axes[index].scatter(xs, ys, color='blue')  # Plot data points

        for i in range(0, len(xs) - 1, 2):
            axes[index].plot(xs[i:i+2], ys[i:i+2], color='red')  # Plot segments

        for i in range(1, len(xs) - 1, 2):
            axes[index].plot(xs[i:i+2], ys[i:i+2], color='blue')  # Plot segments

        axes[index].set_title(str(participant_id))

        axes[index].set_ylim(-1, 11)

        axes[index].set_xlabel('Timestamp', fontsize=18)
        axes[index].set_ylabel('RPE', fontsize=18)
        axes[index].tick_params(axis='both', which='major', labelsize=16)


    # Adjust the spacing between subplots
    plt.tight_layout()

    # # Show all the subplots
    # plt.show()

    plt.savefig("rpe")

if __name__ == "__main__":
    data_dir = "./data"
    output_dir =  "/home/kevin/Documents/quori/src/openface2_ros/rpe"
    offset = 1686000000

    rpe_csv = data_dir + "/Study 1 - Form 2.csv"
    log_files = glob.glob(data_dir + '/logs/**')

    all_timestamps = dict()
    for log_file in log_files:
        process_log(log_file, all_timestamps)

    rpes = process_rpe_csv(rpe_csv)

    get_rpes(all_timestamps, rpes, output_dir)