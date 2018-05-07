fname = '/Volumes/Ext1/Data/AnalysisFingers/08_12_15/roomT8_100_20/script_ramp.08Dec2015_18.45.08.movie'
save_name = strrep(fname, '.movie', '.hdf5')

reader = moviereader(fname);
img = reader.read([1, reader.NumberOfFrames]);
[w, h, tot] = size(img);
save_name = [dname, bn, '.hdf5'];

h5create(save_name,'/data', size(img), ...
    'Datatype','uint16', 'ChunkSize', [w, h, 1] ,'Deflate',5)
h5write(save_name, '/data', img)