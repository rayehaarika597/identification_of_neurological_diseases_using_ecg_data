import pandas as pd
import matplotlib.pyplot as plt
import adi
from decimal import Decimal
import os
import csv

df = pd.read_csv("time_range_team_66 - normal.csv")
sample_frequency = 1000

for row in df.iterrows():
	row = row[1]
	filename = row["ID"]
	file_path = "data/NORMAL/" + filename + ".adicht"
	normal_channel = row['tilt channel']
	if pd.isnull(normal_channel):
		continue
	

	normal_start = row['tilt start ']
	normal_end = row['tilt end']
	if Decimal(normal_start):
		#print('float')
		seconds = round(float(normal_start) - int(normal_start), 1) * 100
		normal_start = int((int(normal_start) * 60) + seconds) * sample_frequency
	else:
		normal_start = int(int(row['tilt start ']) * 60 * sample_frequency)

	if type(normal_end) == float:
		seconds = round(normal_end - int(normal_end), 1) * 100
		normal_end = int((int(normal_end) * 60) + seconds) * sample_frequency
	else:
		normal_end = int(row['tilt end'] * 60 * sample_frequency)


	f = adi.read_file(file_path)
	duration = (normal_end-normal_start) // sample_frequency

	values = f.channels[0].get_data(int(normal_channel))
	ecg_values = values[normal_start:normal_end]

	file_path = os.path.join("sliced_data_tilt/NORMAL/", filename + '.csv')
	
	with open(file_path, mode='w', newline='') as file:
		writer = csv.writer(file)

		for item in ecg_values:
			writer.writerow([item])


	plt.figure(figsize=(20, 7))

	plt.title("ECG signal, slice of %.1f seconds" % duration)
	plt.plot(ecg_values, color="#51A6D8", linewidth=1)
	plt.xlabel("Time (ms)", fontsize=16)
	plt.ylabel("Amplitude (arbitrary unit)")
	plt.savefig("sliced_data_tilt/NORMAL/" + filename + ".png")
	plt.close("all")