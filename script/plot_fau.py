import matplotlib.pyplot as plt

def parse_string(input_str):
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


file_path = "/home/kevin/Documents/quori/src/openface2_ros/processed_data/Participant_8_0/fau.txt"
file = open(file_path, "r")

file_contents = file.read()

time, data = parse_string(file_contents)

# Determine the number of plots and rows needed
num_plots = len(data)
num_rows = num_plots

# Create a new figure and axes for the subplots
fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5*num_rows))

# Iterate over each key-value pair and create a subplot for each
for i, (key, values) in enumerate(data.items()):
    # Plot the values on the current axes
    axes[i].plot(time, values)
    axes[i].set_ylim(0, 5)

    # Set the title of the subplot
    axes[i].set_title(key)

# Adjust the spacing between subplots
plt.tight_layout()

# Show all the subplots
plt.show()
