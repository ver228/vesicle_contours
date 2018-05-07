#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:21:47 2018

@author: avelinojaver
"""

import os
import struct
import math
import numpy as np
import tqdm
#% A Matlab interface to the .movie binary files

#conversion_dict = {'int32':'l', 'uint32':'L', 'uint64':'Q'}


IIDC_header_dtype = np.dtype([
     ('magic', np.uint32),
     ('version', np.uint32),
     ('type', np.uint32),
     ('pixelmode', np.uint32),
     ('length_header', np.uint32),
     ('length_data', np.uint32),
     ('i_guid', np.uint64),
     ('i_vendor_id', np.uint32),
     ('i_model_id', np.uint32),
     ('i_video_mode', np.uint32),
     ('i_color_coding', np.uint32),
     ('i_timestamp_us', np.uint64),
     ('i_size_x_max', np.uint32),
     ('i_size_y_max', np.uint32),
     ('i_size_x', np.uint32),
     ('i_size_y', np.uint32),
     ('i_pos_x', np.uint32),
     ('i_pos_y', np.uint32),
     ('i_pixnum', np.uint32),
     ('i_stride', np.uint32),
     ('i_data_depth', np.uint32),
     ('i_image_bytes', np.uint32),
     ('i_total_bytes', np.uint64),
     ('i_brightness_mode', np.uint32),
     ('i_brightness', np.uint32),
     ('i_exposure_mode', np.uint32),
     ('i_exposure', np.uint32),
     ('i_gamma_mode', np.uint32),
     ('i_gamma', np.uint32),
     ('i_shutter_mode', np.uint32),
     ('i_shutter', np.uint32),
     ('i_gain_mode', np.uint32),
     ('i_gain', np.uint32),
     ('i_temperature_mode', np.uint32),
     ('i_temperature', np.uint32),
     ('i_trigger_delay_mode', np.uint32),
     ('i_trigger_delay', np.uint32),
     ('i_trigger_mode', np.int32),
     ('i_avt_channel_balance_mode', np.uint32),
     ('i_avt_channel_balance', np.int32)
     ])

class MovieReader():
    CAMERA_MOVIE_MAGIC =  b'TemI'
    CAMERA_TYPE_IIDC = 1;
    CAMERA_TYPE_ANDOR= 2;
    CAMERA_TYPE_XIMEA= 3;
    def __init__(self, file_name):
        
        
        dname, fname = os.path.split(file_name)
        
        self.file_name = file_name
        
        self.directory = dname #% The subdirectory that contains the video
        self.moviename = fname  #% The filename of the video
        
        self._read_common_info()

#        self.number_of_frames #% The number of frames contained in the video

#        self._magic;
    
    def _read_common_info(self):
        #% bytes before the data
        word_size = 4
        offset = 0
        with open(self.file_name, 'rb') as fid:
            while True:
                ret = fid.read(word_size)
                offset += 1
                if ret == self.CAMERA_MOVIE_MAGIC:
                    break
                
                if offset >= 1000000:
                    raise('The potential ASCII header is too long. \n No "TemI" found, bailing out. \n Old data type( pre January 2012) ? \n')
        
        self._offset_header =  (offset - 1)*word_size
    
        with open(self.file_name, 'rb') as fid:
            
            self.comments = fid.read((self._offset_header))
            
            
            # Common stuff for all camera types
            common_info = fid.read(24) #%read 24 bytes
            assert common_info[:4] ==  b'TemI'
            common_info = struct.unpack_from("<6L", common_info)
            
            self._version = common_info[1]
            self._camera_type = common_info[2]
            
            
            if self._version == 1:
                self._endian = common_info[3]; #% BigEndian,16bit==2, LittleEndian,16bit==1, 8bit==1, ??12bit==4??
                self._length_header = common_info[4]
                self._length_data = common_info[5]
            elif self.version == 0:
                #%need to hardwire BigEndian/LittleEndian
                self._endian = 2; #% BigEndian==2
                self._length_header = common_info[3]
                self._length_data = common_info[4]
                
            else:
                raise ValueError('Version?')
    
    
        # (44 or 48 bytes from begining of that header to position before c_timestamp)
        timestamp_pos = (24 if self._version == 1 else 20 ) + 24
        with open(self.file_name, 'rb') as fid:
            if self._camera_type == self.CAMERA_TYPE_IIDC:
                #print('IIDC (Grasshopper) camera')
                self._read_iidc(fid, timestamp_pos)
            else:
                raise ValueError('Unimplemented camera!!!')
            
            fid.seek(0, os.SEEK_END)
            file_size = fid.tell()
          
            #%position at the begining of the binary header+frame data
            #fid.seek(self._offset_header , os.SEEK_SET);
           
            nn = (file_size - self._offset_header) / (self._length_header + self._length_data);
            if (math.floor(nn) != nn):
                ValueError('something is wrong in the calculation nn needs to be an integer \n');
            
            
            self.number_of_frames = int(nn);
            self._magic = struct.unpack_from("<L", self.CAMERA_MOVIE_MAGIC)
            
            
    def _read_iidc(self, fid, timestamp_pos) :
        fid.seek(self._offset_header + timestamp_pos, os.SEEK_SET)
        
        ini_fields = 8 if self._version == 0 else 10
        
        
        data_shape = fid.read(8 + ini_fields*4 + 8)
        fmt = '<Q{}LQ'.format(ini_fields)
        data_shape = struct.unpack_from(fmt, data_shape)
        
        
        self._first_frame_timestamp_sec = data_shape[0]/1e6
        
        fid.seek(self._offset_header + self._length_header + self._length_data + timestamp_pos, os.SEEK_SET);
        
        current_ts, = struct.unpack_from('Q', fid.read(8))
        
        self.frame_rate = 1/(current_ts/1e6 - self._first_frame_timestamp_sec)
        self.width = data_shape[3]
        self.height = data_shape[4]
        
        
        self._data_depth = data_shape[9];  #%this is number of bytes 8 or 16
        self._image_bytes = data_shape[10];
        self._total_bytes = data_shape[11];
        
        if (self._data_depth == 16):
            ndiv = 2. #%each word is 2 bytes;
        elif (self._data_depth == 8):
            ndiv = 1.
        elif (self._data_depth == 12):
            ndiv = 1.5 #%each word is an abomination of 12 bits, aka 1.5 bytes;
        else:
            raise ValueError('Bit depth neither 8 nor 12 nor 16 - not programmed for this.\n');
        
        self._length_in_words = math.floor(self._image_bytes/ndiv);
    
    def read_frame(self, frame_number):
        offset = self._offset_header +  frame_number * (self._length_header + self._length_data);
        if self._data_depth == 12:
            raise ValueError('unimplemented')
        
        img_dtype = np.dtype('uint' + str(self._data_depth))
        #field_names, field_types = zip(*IIDC_header)
        #tot_read = sum(size_dict[x] for x in field_types)
        #assert self._length_header == tot_read
        #header_fmt = "<" + ''.join(field_types)
        
        with open(self.file_name, 'rb') as fid:
            fid.seek(offset, os.SEEK_SET)
            header = np.fromfile(fid, count = 1, dtype=IIDC_header_dtype)
            
            #header_bytes = fid.read(self._length_header)
            #header_data = struct.unpack(header_fmt, header_bytes)
            #header = {k:x for k,x in zip(field_names, header_data)}
        
            
            img = np.fromfile(fid, count = self._length_in_words, dtype=img_dtype)
            img = img.reshape((self.height, self.width))
            np.ndarray.byteswap(img, inplace=True)
            
        return header, img

if __name__ == '__main__':
    fname = '/Volumes/Ext1/Data/AnalysisFingers/08_12_15/roomT8_100_20/script_ramp.08Dec2015_18.45.08.movie'
    
    mreader = MovieReader(fname)
    
    

    
    
        
