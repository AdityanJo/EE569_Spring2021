function G = readraw(filename, width, height, bytesperpixel)
	% Get file ID for file
	fid=fopen(filename,'rb');

	% Check if file exists
	if (fid == -1)
	  	error('can not open input image file press CTRL-C to exit \n');
	  	pause
	end

	% Get all the pixels from the image
	pixel = fread(fid, inf, 'uchar');

	% Close file
	fclose(fid);

	% Calculate length/width, assuming image is square
	Size=(height*width*bytesperpixel);

	% Construct matrix
	G = zeros(height,width,bytesperpixel);
    G = reshape(pixel, [width, height, bytesperpixel]);
    G = uint8(G');
end %function