import re
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import csv

def parse_fau(input_str):
    data = {}
    lines = input_str.strip().split('\n')

    time = list(map(float, lines[1].split()))

    for i in range(2, len(lines), 2):
        key_line = lines[i]
        value_line = lines[i + 1]

        key = key_line
        values = list(map(float, value_line.split()))

        data[key] = values

    return time, data


def parse_log(input_str):
    log_lines = input_str.strip().split('\n')

    log_entries = []
    for line in log_lines:
        match = re.match(r'^([\d.]+) - (.*)$', line)
        if match:
            timestamp = float(match.group(1))
            content = match.group(2)
            if '[' in content and ']' in content:
                content = [int(num) for num in re.findall(r'-?\d+', content)]
            log_entries.append((timestamp, content))

    return log_entries





positive_speech = set([
    'Great work, nice pace.', 'Nice job, great speed.',
    'Good speed, keep it up.', 'Nice speed, keep going.',
    'Looks good, great job!', 'Great work, looking good!',
    'Form is good, keep it up.', 'Form looks good, keep going.',
])


data_dir = "/home/kevin/Documents/quori/src/openface2_ros/processed_data"
rpe_dir = "/home/kevin/Documents/quori/src/openface2_ros/rpes"
data_subdirs = glob.glob(data_dir + "/*")

grouped_files = {}

for file_path in data_subdirs:
    # Extract the participant ID from the file path
    participant_id = file_path.split('/')[-1].split('_')[1]
    
    # Check if the participant ID is already in the dictionary
    if participant_id in grouped_files:
        grouped_files[participant_id].append(file_path)
    else:
        grouped_files[participant_id] = [file_path]

max_slope = 0

for part_id in grouped_files:
    if part_id == "12":
        continue

    subdirs = grouped_files[part_id]

    features = defaultdict(defaultdict)
    
    plot_data = []

    for subdir in subdirs:

        print(subdir)

        file_path = subdir + "/fau.txt"
        file = open(file_path, "r")
        file_contents = file.read()
        time, data = parse_fau(file_contents)
        file.close()

        file_path = subdir + "/log.txt"
        file = open(file_path, "r")
        file_contents = file.read()
        log_entries = parse_log(file_contents)
        file.close()

        # Get faus
        for i, timestamp in enumerate(time):
            fau = [value[i] for value in data.values()]
            features[timestamp]["fau"] = fau

        # Get logs
        for entry in log_entries:
            timestamp = float(entry[0])

            closest_timestamp = None
            closest_dist = float("inf")
            for time in features:
                if time <= timestamp and timestamp - time < closest_dist:
                    closest_dist = timestamp - time
                    closest_timestamp = time

            if closest_timestamp == None:
                continue

            if isinstance(entry[1], list):
                features[closest_timestamp]["eval"] = entry[1]
            else:
                if entry[1] in positive_speech:
                    indicator = 1
                else:
                    indicator = -1
                features[closest_timestamp]["speech"] =  indicator

        # Get RPEs
        rpe_file = rpe_dir + "/" + subdir[subdir.find("Participant"): subdir.rfind("_")] + ".txt"
        
        rpe_file_content = open(rpe_file).readlines()
        xs = rpe_file_content[0][:-1].split(", ")
        xs = [float(x) for x in xs]
        ys = rpe_file_content[1][:-1].split(", ")
        ys = [max(1.0, float(y)) for y in ys]

        # plt.scatter(xs, ys)
        # plt.ylim(-1, 11)
        # plt.savefig("og")

        # ts = []
        # rpes = []

        for timestamp in features:
            prev_i = 0
            for i in range(1, len(xs)):
                if xs[i] > timestamp:
                    break
                prev_i = i
            
            delta_x = timestamp - xs[prev_i]

            # exercising
            if prev_i % 2 == 0:
                
                slope = (ys[prev_i + 1] - ys[prev_i]) / (xs[prev_i + 1] - xs[prev_i])
                rpe = delta_x * slope + ys[prev_i]

            # resting
            elif prev_i % 2 == 1:
                slope = -0.041
                rpe = delta_x * slope + ys[prev_i]
                rpe = min(1, rpe)

            features[timestamp]["slope"] = slope
            features[timestamp]["rpe"] = rpe

            max_slope = max(max_slope, slope)
            # ts.append(timestamp)
            # rpes.append(rpe)

            plot_data.append((timestamp, slope))


    # save the feature as a csv
    plot_data.sort()
    ts = [p[0] for p in plot_data]
    ys = [p[1] for p in plot_data]

    output_csv_file = "/home/kevin/Documents/quori/src/openface2_ros/features/" + part_id + ".csv"

    with open(output_csv_file, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["ds", "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU28", "AU45", "E0", "E1", "E2", "E3", "E4", "S", "slope", "rpe"])

        timestamps = list(features.keys())
        timestamps.sort()

        
        for timestamp in timestamps:
            row = []
            row.append(timestamp)
            row.extend(features[timestamp]["fau"])

            if "eval" in features[timestamp]:
                row.extend(features[timestamp]["eval"])
            else:
                row.extend([-2, -2, -2, -2, -2])

            if "speech" in features[timestamp]:
                row.append(features[timestamp]["speech"])
            else:
                row.append(0)
            
            row.append(features[timestamp]["slope"])
            row.append(features[timestamp]["rpe"])
            writer.writerow(row)

    # save the feature as a plot
    plt.scatter(ts, ys)
    plt.ylim(0, 1)
    plt.savefig("/home/kevin/Documents/quori/src/openface2_ros/slope_plots/" + part_id + "_data.png")
    plt.close()

print("max_slope", max_slope)