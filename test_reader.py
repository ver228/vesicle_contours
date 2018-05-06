#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:05:01 2018

@author: ajaver
"""
import struct

#fname = "/Volumes/SanDisk Ultra Fit Media/Cambridge/ramp100.29Oct2015_17.54.52.movie"
fname = '/Users/ajaver/OneDrive - Imperial College London/lucia/ramp40.29Oct2015_18.00.24.movie'

word_size = 4
offset = 0
with open(fname, 'rb') as fid:
    while True:
        try:
            ret = fid.read(word_size)
            offset += 1
            if ret == b'TemI':
                break
            
            
            
        except:
            raise('I was not able to find the magic word in all the file')
            
with open(fname, 'rb') as fid:
    comments = fid.read((offset-1)*word_size)
    #ret = fid.read(word_size)
    #
    
    
    
    
    # Common stuff for all camera types
    common_info = fid.read(6*word_size)
    assert common_info[:word_size] ==  b'TemI'
    common_info = struct.unpack_from("<6L", common_info)
    
    version = common_info[1]
    camera_type = common_info[2]
    
    length_header = common_info[4]
    length_data = common_info[5]
    
    
    
    header = fid.read(length_header*word_size)
    
    dat = struct.unpack_from("<Q8I", header[:8+(4*8)])
    
    #header = struct.unpack_from("<%iL" % length_header, header)
    
    data = fid.read(length_data*word_size)
    data = struct.unpack_from("<%iH" % length_data, data)
    
    
    
#    movie.version = common_info(2);
#    movie.camera_type = common_info(3);
#            
#            if (movie.version == 1)
#                movie.endian=common_info(4); % BigEndian,16bit==2, LittleEndian,16bit==1, 8bit==1, ??12bit==4??
#                movie.length_header = common_info(5);
#                movie.length_data = common_info(6);
#            elseif (movie.version == 0)
#                %need to hardwire BigEndian/LittleEndian
#                movie.endian = 2; % BigEndian==2
#                movie.length_header = common_info(4) ;
#                movie.length_data = common_info(5)  ;
#                fseek(fid,-4,'cof') ; % go back one extra word read into the common_info
#            end
    
    
