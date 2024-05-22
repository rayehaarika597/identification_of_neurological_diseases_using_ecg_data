import pandas as pd
import matplotlib.pyplot as plt
import adi
import os
import csv
import numpy as np

df = pd.read_csv("time_range_team_66 - pots.csv")
sample_frequency = 1000
sequence_length = 299999


for row in df.iterrows():
	row = row[1]
	filename = row["ID"]
	file_path = "data/POTS/" + filename + ".adicht"
	normal_channel = row['channel - lying ']
	if pd.isnull(normal_channel):
		continue
	elif normal_channel == 'data sus':
		continue

	normal_start = row['lying-start']
	normal_end = row['lying-end']
	if type(normal_start) == float:
		seconds = round(normal_start - int(normal_start), 1) * 100
		normal_start = int((int(normal_start) * 60) + seconds) * sample_frequency
	else:
		normal_start = int(row['lying-start'] * 60 * sample_frequency)

	if type(normal_end) == float:
		seconds = round(normal_end - int(normal_end), 1) * 100
		normal_end = int((int(normal_end) * 60) + seconds) * sample_frequency
	else:
		normal_end = int(row['lying-end'] * 60 * sample_frequency)

	f = adi.read_file(file_path)
	duration = (normal_end-normal_start) / sample_frequency

	values = f.channels[0].get_data(int(normal_channel))
	ecg_values = values[normal_start:normal_end]

	# Padding with 0
	if len(ecg_values) < sequence_length:
		padding_length = max(0, sequence_length - len(ecg_values))
		padded_data = np.pad(ecg_values, (0, padding_length), 'constant', constant_values = 0)
	else:
		padded_data = ecg_values[:sequence_length]

	file_path = os.path.join("sliced_data_lying/POTS/", filename + '.csv')
	
	with open(file_path, mode='w', newline='') as file:
		writer = csv.writer(file)

		for item in padded_data:
			writer.writerow([item])


	plt.figure(figsize=(20, 7))

	plt.title("ECG signal, slice of %.1f seconds" % duration)
	plt.plot(padded_data, color="#51A6D8", linewidth=1)
	plt.xlabel("Time (ms)", fontsize=16)
	plt.ylabel("Amplitude (arbitrary unit)")
	plt.savefig("sliced_data_lying/POTS/" + filename + ".png")
	plt.close("all")