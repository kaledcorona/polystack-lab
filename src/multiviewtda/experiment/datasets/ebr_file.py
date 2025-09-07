#------------------------------------------------------------------------------------------------------------------
""" EBR file library
    
    This script contains functions for loading and saving RAW and EBR files. 
    
    A RAW data file is used to store EEG recordings without epoching or processing. These files
    contains the following elements:
        Data type - The data type used to store the EEG record (int, double, float, complex, etc.).
        Number of channels - The number of electrode positions.
        Channel names - The names of the electrodes.
        Number of samples - The number of time points of the EEG record.
        Sampling rate - The sampling rate used to record the EEG data.
        Number of comments - The number of comments added by the experimenter.
        Comment list - The list of comments included in the file.
        Number of marks - The number of marks that indicate time events.
        Mark list - The list of marks that indicate the time events. A mark is a pair that indicates
                    the sameple point of the event and the name of the event.
        Data - The array with the record. Rows represent samples and columns represent channels.
    
    On the other hand, an EBR data file is used to store epoched or band-filtered EEG recordings. 
    These files contains the following elements:
        Data type - The data type used to store the EEG record (int, double, float, complex, etc.).
        Number of trials - The number of trials or repetitions.
        Trial names - The names of the trials.
        Number of channels - The number of electrode positions.
        Channel names - The names of the electrodes.
        Number of bands - The number of bands.
        Band names - The names of the bands.
        Number of samples - The number of time points of the EEG record.
        Sampling rate - The sampling rate used to record the EEG data.
        Number of comments - The number of comments added to the file by the experimenter.
        Comment list - The list of comments included in the file.
        Number of marks - The number of marks that indicate time events.
        Mark list - The list of marks that indicate the time events. A mark is a pair that indicates
                    the sameple point of the event and the name of the event.
        Data - The array with the record. The first dimension represents trials, the second dimension 
               represents channels, the third dimension corresponds to bands, and the last dimension 
               represents the time points.

    This library handles the EBR data files in dictionaries, whose elements represent the data 
    described above.

    Author
    ------
    Omar Mendoza Montoya

    Email
    -----
    omendoz@live.com.mx

    Copyright
    ---------
    Copyright (c) 2022 Omar Mendoza Montoya. All rights reserved.
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,  
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,  
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial 
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
#------------------------------------------------------------------------------------------------------------------

import os
import numpy as np

def ebr_data(trials, channels, bands, samples, samp_rate):
    """ Initialize EBR data structure
        
        This function initializes a EBR data structure for the specified 
        trials, channels, bands, number of samples and sampling rate.

        Parameters
        ----------
        trials : int | list
            The number of trials or a list with the names of the trials. 

        channels : int | list
            The number of channels or a list with the names of the channels. 

        bands : int | list
            The number of bands of the record or a list with the names of the bands.

        samples : int
            The number of samples.

        samp_rate : int | float
            The sampling rate of the data record.

        Returns
        -------
        dict
           A dictionary with the elements specified in the input parameters.
                
    """

    # Check arguments

    if isinstance(trials, int):
        nt = trials
        trial_names = ["Trial " + str(i+1) for i in range(nt)]
    elif isinstance(trials, list) or isinstance(trials, tuple):
        nt = len(trials)
        trial_names = [str(name) for name in trials]
    else:
        raise Exception("The argument 'trials' must be an integer or a list of objects convertible to string.")

    if isinstance(channels, int):
        nc = channels
        channel_names = ["Channel " + str(i+1) for i in range(nc)]
    elif isinstance(channels, list) or isinstance(channels, tuple):
        nc = len(channels)
        channel_names = [str(name) for name in channels]
    else:
        raise Exception("The argument 'channels' must be an integer or a list of objects convertible to string.")

    if isinstance(bands, int):
        nb = bands
        band_names = ["Band " + str(i+1) for i in range(nb)]
    elif isinstance(bands, list) or isinstance(bands, tuple):
        nb = len(bands)
        band_names = [str(name) for name in bands]
    else:
        raise Exception("The argument 'bands' must be an integer or a list of objects convertible to string.")

    if isinstance(samples, int):
        ns = samples
    else:
        raise Exception("The argument 'samples' must be an integer.")

    if isinstance(samp_rate, int) or isinstance(samp_rate, float):
        fs = float(samp_rate)
    else:
        raise Exception("The argument 'samples' must be numeric.")

    # Initialize data fields
    data_type = 'double'

    ncomments = 1
    comments = ["EEG Data"]

    nmarks = 1
    marks = [(0, "origin")]

    data = np.zeros([nt, nc, nb, ns])

    # Build data structure
    ebr_data = dict()
    
    ebr_data["data_type"] = data_type

    ebr_data["sampling_rate"] = fs

    ebr_data["number_of_trials"] = nt
    ebr_data["trials"] = trial_names

    ebr_data["number_of_channels"] = nc
    ebr_data["channels"] = channel_names

    ebr_data["number_of_bands"] = nb
    ebr_data["bands"] = band_names

    ebr_data["number_of_samples"] = ns
    
    ebr_data["number_of_comments"] = ncomments
    ebr_data["comments"] = comments

    ebr_data["number_of_marks"] = nmarks
    ebr_data["marks"] = marks

    ebr_data["data"] = data

    return ebr_data

def save_ebr_file(file, data):
    """ Save EBR file
        
        This function saves an EBR data structure in a file.

        Parameters
        ----------
        file : str
            The name of the file to create with the data.

        data : dict
            The data structure with the data to save.           
    """

    # Check arguments
    if not isinstance(file, str):        
        raise Exception("The argument 'file' must be a string.")

    file_path = os.path.dirname(file)
    if file_path!='' and not os.path.exists(file_path):
        raise Exception("The specified path is not valid or does not exit.")        

    # Open file
    data_file = open(file, "wb")

    # Save header
    data_file.write(b'ebr binary 1.0\n')

    data_file.write(('data_type ' + data['data_type'] + '\n').encode(encoding = 'UTF-8'))

    data_file.write(('sampling_rate ' + str(data['sampling_rate']) +'\n').encode(encoding = 'UTF-8'))
    data_file.write(('samples ' + str(data['number_of_samples']) +'\n').encode(encoding = 'UTF-8'))
    
    data_file.write(('bands ' + str(data['number_of_bands']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_bands']):
        data_file.write(('\tband_' + str(i+1) + ' ' + str(data['bands'][i]) +'\n').encode(encoding = 'UTF-8'))

    data_file.write(('channels ' + str(data['number_of_channels']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_channels']):
        data_file.write(('\tchannel_' + str(i+1) + ' ' + str(data['channels'][i]) +'\n').encode(encoding = 'UTF-8'))

    data_file.write(('trials ' + str(data['number_of_trials']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_trials']):
        data_file.write(('\ttrial_' + str(i+1) + ' ' + str(data['trials'][i]) +'\n').encode(encoding = 'UTF-8'))

    data_file.write(('comments ' + str(data['number_of_comments']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_comments']):
        data_file.write(('\tcomment_' + str(i+1) + ' ' + str(data['comments'][i]) +'\n').encode(encoding = 'UTF-8'))


    data_file.write(('marks ' + str(data['number_of_marks']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_marks']):
        data_file.write(('\tmark_' + str(i+1) + ' ' + str(data['marks'][i][0]) + ' ' + str(data['marks'][i][1])  +'\n').encode(encoding = 'UTF-8'))

    data_file.write(b'end_header\n')

    # Save data 
    data_type =  data['data_type']

    if data_type == 'int8' or data_type== 'char':
        data_file.write(data['data'].astype(np.int8).tobytes())

    elif data_type == 'uint8' or data_type== 'unsigned char':
        data_file.write(data['data'].astype(np.uint8).tobytes())

    elif data_type == 'int16' or data_type== 'short':
        data_file.write(data['data'].astype(np.int16).tobytes())

    elif data_type == 'uint16' or data_type== 'unsigned short':
        data_file.write(data['data'].astype(np.uint16).tobytes())

    elif data_type == 'int32' or data_type== 'int':
        data_file.write(data['data'].astype(np.int32).tobytes())

    elif data_type == 'uint32' or data_type== 'unsigned int':
        data_file.write(data['data'].astype(np.uint32).tobytes())

    elif data_type == 'int64' or data_type== '__int64':
        data_file.write(data['data'].astype(np.int64).tobytes())

    elif data_type == 'uint64' or data_type== 'unsigned __int64':
        data_file.write(data['data'].astype(np.uint64).tobytes())

    elif data_type == 'float':
        data_file.write(data['data'].astype(np.float32).tobytes())

    elif data_type == 'double':
        data_file.write(data['data'].astype(np.float64).tobytes())

    elif data_type == 'complex' or data_type== 'class std::complex<double>':
        data_file.write(data['data'].astype(np.cdouble).tobytes()) 

    data_file.close()
    return


def load_ebr_file(file):
    """ Load EBR file
        
        This function loads an EBR file and returns its content in a dictionary.

        Parameters
        ----------
        file : str
            The name of the file to load.

        Returns
        -------
        dict
           A dictionary with the loaded data.                
    """

    # Check arguments

    if not isinstance(file, str):        
        raise Exception("The argument 'file' must be a string.")

    if not os.path.exists(file):
        raise Exception("The specified path is not valid or does not exit.")        

    # Open file
    data_file = open(file, "rb")

    # Read magic key
    magic = data_file.readline().strip().lower()
    if not magic == b'ebr binary 1.0':
        raise Exception("The specified file is not a binary EBR file.")

    # Read header
    data_type = "double"
    fs = 0
    ns = 0
    nb = 0
    bands = []
    nc = 0
    channels = []
    nt = 0
    trials = []    
    ncomments = 0
    comments = []
    nmarks = 0
    marks = []

    while True:
        line = data_file.readline().strip()
       
        if line.startswith(b'data_type'):
            splited_line = line.split(b'data_type', 1)
            data_type = splited_line[1].strip().decode("utf-8") 

        elif line.startswith(b'sampling_rate'):
            splited_line = line.split(b'sampling_rate', 1)
            fs = float(splited_line[1].strip())

        elif line.startswith(b'samples'):
            splited_line = line.split(b'samples', 1)
            ns = int(splited_line[1].strip())

        elif line.startswith(b'bands'):
            splited_line = line.split(b'bands', 1)
            nb = int(splited_line[1].strip())
            bands = ['']*nb

        elif line.startswith(b'band_'):
            splited_line = line.split(b'band_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            bands[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'channels'):
            splited_line = line.split(b'channels', 1)
            nc = int(splited_line[1].strip())
            channels = ['']*nc

        elif line.startswith(b'channel_'):
            splited_line = line.split(b'channel_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            channels[index] = info[1].strip().decode("utf-8")  
            
        elif line.startswith(b'trials'):
            splited_line = line.split(b'trials', 1)
            nt = int(splited_line[1].strip())
            trials = ['']*nt

        elif line.startswith(b'trial_'):
            splited_line = line.split(b'trial_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            trials[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'comments'):
            splited_line = line.split(b'comments', 1)
            ncomments = int(splited_line[1].strip())
            comments = ['']*ncomments

        elif line.startswith(b'comment_'):
            splited_line = line.split(b'comment_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            comments[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'marks'):
            splited_line = line.split(b'marks', 1)
            nmarks = int(splited_line[1].strip())
            marks = ['']*nmarks

        elif line.startswith(b'mark_'):
            splited_line = line.split(b'mark_', 1)
            info = splited_line[1].split(b' ', 2)
            index = int(info[0])-1
            mark_index = int(info[1])
            marks[index] = (mark_index, info[2].strip().decode("utf-8"))

        elif line.startswith(b'end_header'):
            break

    # Read data
    data_size = nt*nc*nb*ns
     
    if data_type == 'int8' or data_type== 'char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int8))
        data = data.astype('float64')

    elif data_type == 'uint8' or data_type== 'unsigned char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint8))
        data = data.astype('float64')

    elif data_type == 'int16' or data_type== 'short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int16))
        data = data.astype('float64')

    elif data_type == 'uint16' or data_type== 'unsigned short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint16))
        data = data.astype('float64')

    elif data_type == 'int32' or data_type== 'int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int32))
        data = data.astype('float64')

    elif data_type == 'uint32' or data_type== 'unsigned int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint32))
        data = data.astype('float64')

    elif data_type == 'int64' or data_type== '__int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int64))
        data = data.astype('float64')

    elif data_type == 'uint64' or data_type== 'unsigned __int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint64))
        data = data.astype('float64')

    elif data_type == 'float':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float32))
        data = data.astype('float64')

    elif data_type == 'double':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float64))

    elif data_type == 'complex' or data_type== 'class std::complex<double>':
        raw_data = data_file.read(16*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.cdouble))

    data = data.reshape((nt, nc, nb, ns))

    data_file.close()

    # Build data structure
    ebr_data = dict()
    
    ebr_data["data_type"] = data_type

    ebr_data["sampling_rate"] = fs

    ebr_data["number_of_trials"] = nt
    ebr_data["trials"] = trials

    ebr_data["number_of_channels"] = nc
    ebr_data["channels"] = channels

    ebr_data["number_of_bands"] = nb
    ebr_data["bands"] = bands

    ebr_data["number_of_samples"] = ns
    
    ebr_data["number_of_comments"] = ncomments
    ebr_data["comments"] = comments

    ebr_data["number_of_marks"] = nmarks
    ebr_data["marks"] = marks

    ebr_data["data"] = data

    return ebr_data


def raw_data(channels, samples, samp_rate):
    """ Initialize RAW data structure
        
        This function initializes a RAW data structure for the specified 
        channels, samples and sampling rate.

        Parameters
        ----------
        channels : int | list
            The number of channels or a list with the names of the channels.         

        samples : int
            The number of samples.

        samp_rate : int | float
            The sampling rate of the data record.

        Returns
        -------
        dict
           A dictionary with the elements specified in the input parameters.
                
    """

    # Check arguments

    if isinstance(channels, int):
        nc = channels
        channel_names = ["Channel " + str(i+1) for i in range(nc)]
    elif isinstance(channels, list) or isinstance(channels, tuple):
        nc = len(channels)
        channel_names = [str(name) for name in channels]
    else:
        raise Exception("The argument 'channels' must be an integer or a list of objects convertible to string.")

    if isinstance(samples, int):
        ns = samples
    else:
        raise Exception("The argument 'samples' must be an integer.")

    if isinstance(samp_rate, int) or isinstance(samp_rate, float):
        fs = float(samp_rate)
    else:
        raise Exception("The argument 'samples' must be numeric.")

    # Initialize data fields
    data_type = 'double'

    ncomments = 1
    comments = ["EEG Data"]

    nmarks = 1
    marks = [(0, "origin")]

    data = np.zeros([nc, ns])

    # Build data structure
    ebr_data = dict()
    
    ebr_data["data_type"] = data_type

    ebr_data["sampling_rate"] = fs

    ebr_data["number_of_channels"] = nc
    ebr_data["channels"] = channel_names

    ebr_data["number_of_samples"] = ns
    
    ebr_data["number_of_comments"] = ncomments
    ebr_data["comments"] = comments

    ebr_data["number_of_marks"] = nmarks
    ebr_data["marks"] = marks

    ebr_data["data"] = data

    return ebr_data

def save_raw_file(file, data):
    """ Save RAW file
        
        This function saves a RAW data structure in a file.

        Parameters
        ----------
        file : str
            The name of the file to create with the data.

        data : dict
            The data structure with the data to save.           
    """

    # Check arguments
    if not isinstance(file, str):        
        raise Exception("The argument 'file' must be a string.")

    file_path = os.path.dirname(file)
    if file_path!='' and not os.path.exists(file_path):
        raise Exception("The specified path is not valid or does not exit.")        

    # Open file
    data_file = open(file, "wb")

    # Save header
    data_file.write(b'raw binary 1.0\n')

    data_file.write(('data_type ' + data['data_type'] + '\n').encode(encoding = 'UTF-8'))

    data_file.write(('sampling_rate ' + str(data['sampling_rate']) +'\n').encode(encoding = 'UTF-8'))
    data_file.write(('samples ' + str(data['number_of_samples']) +'\n').encode(encoding = 'UTF-8'))
    
    data_file.write(('channels ' + str(data['number_of_channels']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_channels']):
        data_file.write(('\tchannel_' + str(i+1) + ' ' + str(data['channels'][i]) +'\n').encode(encoding = 'UTF-8'))

    data_file.write(('comments ' + str(data['number_of_comments']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_comments']):
        data_file.write(('\tcomment_' + str(i+1) + ' ' + str(data['comments'][i]) +'\n').encode(encoding = 'UTF-8'))

    data_file.write(('marks ' + str(data['number_of_marks']) +'\n').encode(encoding = 'UTF-8'))
    for i in range(data['number_of_marks']):
        data_file.write(('\tmark_' + str(i+1) + ' ' + str(data['marks'][i][0]) + ' ' + str(data['marks'][i][1])  +'\n').encode(encoding = 'UTF-8'))

    data_file.write(b'end_header\n')

    # Save data 
    data_type =  data['data_type']

    if data_type == 'int8' or data_type== 'char':
        data_file.write(data['data'].astype(np.int8).transpose().tobytes())

    elif data_type == 'uint8' or data_type== 'unsigned char':
        data_file.write(data['data'].astype(np.uint8).transpose().tobytes())

    elif data_type == 'int16' or data_type== 'short':
        data_file.write(data['data'].astype(np.int16).transpose().tobytes())

    elif data_type == 'uint16' or data_type== 'unsigned short':
        data_file.write(data['data'].astype(np.uint16).transpose().tobytes())

    elif data_type == 'int32' or data_type== 'int':
        data_file.write(data['data'].astype(np.int32).transpose().tobytes())

    elif data_type == 'uint32' or data_type== 'unsigned int':
        data_file.write(data['data'].astype(np.uint32).transpose().tobytes())

    elif data_type == 'int64' or data_type== '__int64':
        data_file.write(data['data'].astype(np.int64).transpose().tobytes())

    elif data_type == 'uint64' or data_type== 'unsigned __int64':
        data_file.write(data['data'].astype(np.uint64).transpose().tobytes())

    elif data_type == 'float':
        data_file.write(data['data'].astype(np.float32).transpose().tobytes())

    elif data_type == 'double':
        data_file.write(data['data'].astype(np.float64).transpose().tobytes())

    elif data_type == 'complex' or data_type== 'class std::complex<double>':
        data_file.write(data['data'].astype(np.cdouble).transpose().tobytes()) 

    data_file.close()
    return


def load_raw_file(file):
    """ Load RAW file
        
        This function loads a RAW file and returns its content in a dictionary.

        Parameters
        ----------
        file : str
            The name of the file to load.

        Returns
        -------
        dict
           A dictionary with the loaded data.                
    """

    # Check arguments

    if not isinstance(file, str):        
        raise Exception("The argument 'file' must be a string.")

    if not os.path.exists(file):
        raise Exception("The specified path is not valid or does not exit.")        

    # Open file
    data_file = open(file, "rb")

    # Read magic key
    magic = data_file.readline().strip().lower()
    if not magic == b'raw binary 1.0':
        raise Exception("The specified file is not a binary RAW file.")

    # Read header
    data_type = "double"
    fs = 0
    ns = 0
    nc = 0
    channels = []
    ncomments = 0
    comments = []
    nmarks = 0
    marks = []

    while True:
        line = data_file.readline().strip()
       
        if line.startswith(b'data_type'):
            splited_line = line.split(b'data_type', 1)
            data_type = splited_line[1].strip().decode("utf-8") 

        elif line.startswith(b'sampling_rate'):
            splited_line = line.split(b'sampling_rate', 1)
            fs = float(splited_line[1].strip())

        elif line.startswith(b'samples'):
            splited_line = line.split(b'samples', 1)
            ns = int(splited_line[1].strip())

        elif line.startswith(b'channels'):
            splited_line = line.split(b'channels', 1)
            nc = int(splited_line[1].strip())
            channels = ['']*nc

        elif line.startswith(b'channel_'):
            splited_line = line.split(b'channel_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            channels[index] = info[1].strip().decode("utf-8")  
            
        elif line.startswith(b'comments'):
            splited_line = line.split(b'comments', 1)
            ncomments = int(splited_line[1].strip())
            comments = ['']*ncomments

        elif line.startswith(b'comment_'):
            splited_line = line.split(b'comment_', 1)
            info = splited_line[1].split(b' ', 1)
            index = int(info[0])-1
            comments[index] = info[1].strip().decode("utf-8")  

        elif line.startswith(b'marks'):
            splited_line = line.split(b'marks', 1)
            nmarks = int(splited_line[1].strip())
            marks = ['']*nmarks

        elif line.startswith(b'mark_'):
            splited_line = line.split(b'mark_', 1)
            info = splited_line[1].split(b' ', 2)
            index = int(info[0])-1
            mark_index = int(info[1])
            marks[index] = (mark_index, info[2].strip().decode("utf-8"))

        elif line.startswith(b'end_header'):
            break

    # Read data
    data_size = nc*ns
     
    if data_type == 'int8' or data_type== 'char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int8))
        data = data.astype('float64')

    elif data_type == 'uint8' or data_type== 'unsigned char':
        raw_data = data_file.read(data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint8))
        data = data.astype('float64')

    elif data_type == 'int16' or data_type== 'short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int16))
        data = data.astype('float64')

    elif data_type == 'uint16' or data_type== 'unsigned short':
        raw_data = data_file.read(2*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint16))
        data = data.astype('float64')

    elif data_type == 'int32' or data_type== 'int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int32))
        data = data.astype('float64')

    elif data_type == 'uint32' or data_type== 'unsigned int':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint32))
        data = data.astype('float64')

    elif data_type == 'int64' or data_type== '__int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.int64))
        data = data.astype('float64')

    elif data_type == 'uint64' or data_type== 'unsigned __int64':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.uint64))
        data = data.astype('float64')

    elif data_type == 'float':
        raw_data = data_file.read(4*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float32))
        data = data.astype('float64')

    elif data_type == 'double':
        raw_data = data_file.read(8*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.float64))

    elif data_type == 'complex' or data_type== 'class std::complex<double>':
        raw_data = data_file.read(16*data_size)
        data = np.frombuffer(raw_data, dtype=np.dtype(np.cdouble))

    data = data.reshape((ns, nc)).transpose()

    data_file.close()

    # Build data structure

    ebr_data = dict()
    
    ebr_data["data_type"] = data_type

    ebr_data["sampling_rate"] = fs

    ebr_data["number_of_channels"] = nc
    ebr_data["channels"] = channels

    ebr_data["number_of_samples"] = ns
    
    ebr_data["number_of_comments"] = ncomments
    ebr_data["comments"] = comments

    ebr_data["number_of_marks"] = nmarks
    ebr_data["marks"] = marks

    ebr_data["data"] = data

    return ebr_data

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
