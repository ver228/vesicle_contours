addpath('/Users/ajaver/OneDrive - Imperial College London/lucia/MATLABscripts/ReaderFunctions')

dname = '/Users/ajaver/OneDrive - Imperial College London/lucia/';

bn = 'ramp40.29Oct2015_18.00.24';
%bn = 'ramp100.29Oct2015_17.54.52';

fname = [dname, bn, '.movie'];
reader = moviereader(fname);

img = reader.read([1,reader.NumberOfFrames]);
%%
[w, h, tot] = size(img);

save_name = [dname, bn, '.hdf5'];

h5create(save_name,'/data', size(img), ...
    'Datatype','uint16', 'ChunkSize', [w, h, 1] ,'Deflate',5)

h5write(save_name, '/data', img)