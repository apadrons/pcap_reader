from scapy.all import rdpcap, IP, UDP
import datetime
import os
import math
import numpy as np
import laspy
import pandas as pd

def horizontal_angle(row):
    channel_number,azimuth_block,rpm = row['Channel'],row['Azimuth'],row['RPM']
    horizontal_angle_offset = 0

    #firing_time_channel = absolute_time +5.632-50 * (8-block_number)
    firing_time_offset = ((1.512*(channel_number -1) +.368)/1000000)


    angle_block = (azimuth_block * math.pi/180) + horizontal_angle_offset
    firing_time_offset = firing_time_offset * (rpm*6)
    result_horizontal_angle = angle_block + firing_time_offset

    return result_horizontal_angle


def vertical_angle (row):
    channel = row['Channel']
    angle = 15 - (channel-1)
    angle = angle * math.pi/180

    return angle

def convert_x(row):
    distance = row['Distance'] * 0.004
    angle_vertical = row['Vertical Angle']
    angle_horizontal = row['Horizontal Angle']

    x = distance * math.cos(angle_vertical) * math.sin(angle_horizontal)
    return x

def convert_y(row):
    distance = row['Distance'] * 0.004
    angle_vertical = row['Vertical Angle']
    angle_horizontal = row['Horizontal Angle']

    y = distance *  math.cos(angle_vertical) * math.cos(angle_horizontal)
    return y

def convert_z(row):
    distance = row['Distance'] * 0.004
    angle_horizontal = row['Horizontal Angle']
    angle_vertical = row['Vertical Angle']

    z = distance * math.sin(angle_vertical)
    return z

def angle_convertion (df):
    df['Horizontal Angle'] = df.apply(horizontal_angle,axis = 1)
    df['Vertical Angle'] = df.apply(vertical_angle,axis = 1)

def coordinate_correction (df):
    df['x'] = df.apply(convert_x, axis = 1)
    df['y'] = df.apply(convert_y, axis = 1)
    df['z'] = df.apply(convert_z, axis = 1)

def extract_and_create_table(pcap_folder):
    calibration_df = pd.read_csv('hesai_calibration.csv')
    file_list = os.listdir(pcap_folder)
    # print(file_list)
    frame_counter = 1
    csv_counter = 0
    for file in file_list[::100]:
        # print(os.path.join(pcap_folder,file))
        packets = rdpcap(os.path.join(pcap_folder,file))
        data_list = []

        for packet in packets:
            if UDP in packet:
                raw_data = bytes(packet[UDP].payload)


                #extract pre-header,header,body
                pre_header = raw_data[:6]
                header = raw_data[6:12]
                body = raw_data[12:-28]
                tail = raw_data[-28:]

                #****************Tail data ******************
                return_mode = tail[10]
                rpm = int.from_bytes(tail[11:13],byteorder = 'little')
                date_time = tail[13:19]
                if date_time[1] < 13 and date_time[1] > 0: #Check the range for month
                    year = date_time[0] + 1900
                    dt = datetime.datetime(year,date_time[1],date_time[2],date_time[3],date_time[4],date_time[5])
                    timestamp = dt.timestamp() - 14400 #To adjust GPS it needs to be synchronized every 4 hours
                    micro_seconds = int.from_bytes(tail[19:23],byteorder='little')
                    absolute_time = timestamp + (micro_seconds/1000000)


                #*********************************
                body_length = len(body)
                block_size = 130 #Each block has 130 bytes
                num_blocks = body_length//block_size



                for i in range(1,num_blocks):
                    block = body[i*block_size:(i+1)*block_size]
                    azimuth = int.from_bytes(block[:2], byteorder='little')
                    azimuth = azimuth * 0.01
                    if azimuth == 0:
                        frame_counter += 1
                    # print(f'Azimuth: {azimuth} degrees')

                    for channel in range(32):
                        offset = 2 + channel * 4
                        distance = int.from_bytes(block[offset:offset+2],byteorder='little')
                        reflectivity = block[offset+2]
                        reserved = block[offset+3]

                        # print(f'Channel {channel+1}: Distance {distance}, Reflectivity {reflectivity}')
                        #Obtaining corrected angle with calibration file
                        azimuth_correction = calibration_df.at[channel,'Azimuth']

                        data_list.append(
                            {
                                "Frame":frame_counter,
                                "Block Number":i+1,
                                'Channel':channel+1,
                                'Azimuth':azimuth + azimuth_correction,
                                'Distance':distance,
                                'Reflectivity':reflectivity,
                                'Unix time':timestamp,
                                'Microseconds':micro_seconds,
                                'Absolute Time':absolute_time,
                                'RPM':rpm

                            }
                        )

    # Create a DataFrame
        df = pd.DataFrame(data_list)
        file_name = f'extracted_data_{csv_counter}.csv'
        df.to_csv(os.path.join(pcap_folder,file_name), index=False)
        csv_counter += 1






# Function to apply the rotation to a DataFrame
def rotate_points_df(df, rotation_matrix):
    # Convert the DataFrame to a NumPy array
    points = df[['x', 'y', 'z']].values
    # Apply the rotation
    rotated_points = np.dot(points, rotation_matrix.T)
    # Return a new DataFrame with rotated points
    df[['x', 'y', 'z']] = rotated_points
    return df


def rotate_z_point(x, y, z, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    point = np.array([x, y, z])
    rotated_point = rotation_matrix.dot(point)
    return rotated_point


def df_to_las(df,file_name):
    header = laspy.LasHeader(point_format=3,version='1.2')
    header.x_scale =  0.01
    header.y_scale =  0.01
    header.z_scale =  0.01

    las = laspy.LasData(header)
    # las.intensity = df['Reflectivity'].to_numpy()
    las.x = df['x'].to_numpy()
    las.y = df['y'].to_numpy()
    las.z = df['z'].to_numpy()

    las.write(f"{file_name}.las")

''' This section extracts the pcap packets and converts the data in csv file in the target folder path '''
#
# targetfolderpath="I:/20240803_092316_app_run/20240803_100158_Data_stream/broken pcap/"
# extract_and_create_table(targetfolderpath)

''' This section generates the points based on the angles and distance. '''
file_path = 'I:/20240803_092316_app_run/20240803_100158_Data_stream/broken pcap/'
csv_files = os.listdir(file_path)
csv_files =[os.path.join(file_path,file) for file in csv_files if file.endswith('.csv')]

file_count = 0

for file in csv_files:
    print(f"Working on file number=> {file_count}")
    df = pd.read_csv(file)
    initial_df = df
    angle_convertion(initial_df)
    coordinate_correction(initial_df)
    initial_df = initial_df[(initial_df['x'] != 0) | (initial_df['y'] != 0) | (initial_df['z'] != 0)]
    initial_df.to_csv('cleaned_data.csv')

    ''' This section rotates all the points around the x and y axis. '''
    # Convert degrees to radians
    df = pd.read_csv('cleaned_data.csv')
    theta_x = np.radians(-30)
    theta_y = np.radians(20)
    #
    # Rotation matrix around x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    # Rotation matrix around y-axis
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # Combined rotation matrix
    R_combined = np.dot(R_y, R_x)
    # Apply the rotation to the DataFrame
    df_rotated = rotate_points_df(df, R_combined)
    df_rotated = df_rotated[['Reflectivity', 'Unix time', 'Microseconds',
                             'Absolute Time', 'x', 'y', 'z']]
    df_rotated.to_csv('rotated_data.csv')

    ''' This section cleans and merges the GPS data with the new lidar data '''
    df_gps = pd.read_csv('Cleaned_state.csv')
    df_lidar = pd.read_csv('rotated_data.csv')
    df_time = pd.DataFrame()
    df_lidar.rename(columns={'Unix time': 'Unix Time'}, inplace=True)
    df_gps['Absolute Time'] = df_gps.apply(lambda row: row['Unix Time'] + (row['Microseconds'] / 1000000), axis=1)
    df_merged = pd.merge(df_lidar, df_gps, on='Unix Time', how='left', suffixes=('_lidar', '_gps'))
    # print(df_merged.head(5))
    df_merged = df_merged[['Reflectivity', 'Unix Time', 'Microseconds_lidar',
                           'x', 'y', 'z', 'Microseconds_gps', 'easting',
                           'northing', 'Heading (degrees)']]
    df_merged = df_merged.drop_duplicates(subset=['x', 'y', 'z'], keep='last')
    df_merged.to_csv('temp.csv')  # Temp to check northing and easting after merge

    points = df_merged[['x', 'y', 'z']].values
    angle = np.radians(df_merged['Heading (degrees)'].values)
    # print(df_merged[['easting','x','northing','y']])
    ''' Loop to rotate Data around the z-axis to match heading '''

    rotated_points = []

    for index, row in df_merged.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        angle = row['Heading (degrees)'] - 15
        rotated_point = rotate_z_point(x, y, z, angle).tolist()
        '''Correcting based on the gps after rotating'''

        df_merged.at[index,'x'] = row['easting'] - rotated_point[0]
        df_merged.at[index,'y'] = row['northing'] - rotated_point[1]
        df_merged.at[index,'z'] = 2 + rotated_point[2]


    df_merged.to_csv('merged_gps_lidar.csv')
    # print(df_merged[['x','y']].values)

    ''' This section generates the LAS file '''
    df = pd.read_csv('merged_gps_lidar.csv')
    # df = df[df['Unix Time'] == 1722679318]
    file_name = f'Result_{file_count}'
    file_count+=1
    df_to_las(df,file_name)


