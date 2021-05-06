/************************************
 *  Name: Zongjian Li               *
 *  USC ID: 6503378943              *
 *  USC Email: zongjian@usc.edu     *
 *  Submission Date: 3rd,Mar 2019   *
 ************************************/
// DO NOT SUBMIT THIS PLZ :/ 
 /*=================================
 |                                 |
 |              util               |
 |                                 |
 =================================*/

 //#define  _CRT_SECURE_NO_WARNINGS 

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <tuple>
#include <stack>

using namespace std;

const char * ArgumentOutOfRangeException = "Argument out of range";

const char * ArgumentException = "Wrong argument";

const char * InvalidOperationException = "Invalid operation";

const char * FailedToOpenFileException = "Failed to open file";

const char * DEFAULT_OUTPUT_FILENAME = "output.raw";

const int UNSIGNED_CHAR_MAX_VALUE = 0xFF;

const int GRAY_CHANNELS = 1;

const int COLOR_CHANNELS = 3;

enum class PaddingType {
	Zero,
	Reflect,
};

template <typename T>
class Image {
	private:

	protected:
	int Height;
	int Width;
	int Channel;
	vector<T> Data;
	PaddingType Padding;

	int getSize() const {
		return Height * Width * Channel;
	}

	int getMemSize() const {
		return sizeof(unsigned char) * getSize();
	}

	static int reflect(const int size, const int index) {
		if (index <= -size || index >= size * 2 - 1) {
			throw ArgumentOutOfRangeException;
		}
		if (index < 0) {
			return -index;
		} else if (index >= size) {
			return 2 * size - index - 2;
		} else {
			return index;
		}
	}

	int index(const int row, const int col, const int ch) const {
		return Channel * (reflect(Height, row) * Width + reflect(Width, col)) + ch;
	}

	public:
	Image() : Height(0), Width(0), Channel(1), Padding(PaddingType::Reflect) {
		Data = vector<T>();
	}

	Image(const int height, const int width, const int channel, const PaddingType padding = PaddingType::Reflect) : Height(height), Width(width), Channel(channel), Padding(padding) {
		Data = vector<T>(getSize());
	}

	Image(const int height, const int width, const int channel, const T & fill) : Image(height, width, channel) {
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			Data[i] = fill;
		}
	}

	Image(const int height, const int width, const int channel, const T data[]) : Image(height, width, channel) {
		auto size = getSize();
		for (int i = 0; i < size; i++) {
			Data[i] = data[i];
		}
	}

	Image(const int height, const int width, const int channel, string filename) : Image(height, width, channel) {
		auto s = ifstream(filename, ifstream::binary);
		if (!s) {
			throw FailedToOpenFileException;
		}
		s.read((char *)&(Data[0]), getMemSize());
		s.close();
	}

	Image(const vector<Image> & channels) {
		if (channels.empty()) {
			throw ArgumentOutOfRangeException;
		}
		auto sample = channels.front();
		for (auto & ch : channels) {
			if (ch.Channel != 1) {
				throw ArgumentException;
			}
			if (ch.Width != sample.Width || ch.Height != sample.Height) {
				throw ArgumentException;
			}
		}
		Height = sample.Height;
		Width = sample.Width;
		Channel = channels.size();
		Data = vector<T>(getSize());
		Padding = PaddingType::Reflect;
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width; j++) {
				for (int ch = 0; ch < Channel; ch++) {
					setValue(i, j, ch, channels[ch].getValue(i, j, 0));
				}
			}
		}
	}

	int getHeight() const {
		return Height;
	}

	int getWidth() const {
		return Width;
	}

	int getChannel() const {
		return Channel;
	}

	T getValue(const int row, const int col, const int ch, const bool enablePadding = true) const {
		if (row < 0 || row >= Height || col < 0 || col >= Width || ch < 0 || ch >= Channel) {
			if (!enablePadding) {
				throw ArgumentOutOfRangeException;
			} else if (Padding == PaddingType::Zero) {
				return 0;
			}
		}
		return Data[index(row, col, ch)];
	}

	void setValue(const int row, const int col, const int ch, const T & value) {
		if (row < 0 || row >= Height || col < 0 || col >= Width || ch < 0 || ch >= Channel) {
			throw ArgumentOutOfRangeException;
		}
		Data[index(row, col, ch)] = value;
	}

	PaddingType getPadding() const {
		return Padding;
	}

	void setPadding(const PaddingType padding) {
		Padding = padding;
	}

	Image clip(const T lower, const T upper) const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = min(upper, max(lower, Data[i]));
		}
	}

	Image round() const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = int(Data[i] + 0.5);
		}
	}

	Image comp(const T upper = 0) const {
		auto result = Image(Height, Width, Channel);
		for (int i = 0; i < getSize(); i++) {
			result.Data[i] = upper - Data[i];
		}
		return result;
	}

	vector<Image> split() const {
		auto result = vector<Image>(Channel);
		for (int ch = 0; ch < Channel; ch++) {
			result[ch] = Image(Height, Width, 1);
			for (int i = 0; i < Height; i++) {
				for (int j = 0; j < Width; j++) {
					result[ch].setValue(i, j, 0, getValue(i, j, ch));
				}
			}
		}
		return result;
	}

	void writeToFile(const string & filename) const {
		auto s = ofstream(filename, ofstream::binary);
		if (!s) {
			throw FailedToOpenFileException;
		}
		s.write((char *)&Data[0], getMemSize());
		s.close();
	}

	~Image() {

	}
};

const float THRESHOLD = 0.5 * UNSIGNED_CHAR_MAX_VALUE;

const int DEFAULT_HEIGHT = 247;

const int DEFAULT_WIDTH = 247;

/*=================================
|                                 |
|                a)               |
|                                 |
=================================*/

Image<bool> Binarize(const Image<unsigned char> & input) {
	if (input.getChannel() != GRAY_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<bool>(input.getHeight(), input.getWidth(), GRAY_CHANNELS, PaddingType::Zero);
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto v = input.getValue(i, j, 0);
			result.setValue(i, j, 0, v > THRESHOLD);
		}
	}
	return result;
}

Image<unsigned char> Strech(const Image<bool> & input) {
	if (input.getChannel() != GRAY_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<unsigned char>(input.getHeight(), input.getWidth(), GRAY_CHANNELS, PaddingType::Zero);
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto v = input.getValue(i, j, 0);
			result.setValue(i, j, 0, v ? UNSIGNED_CHAR_MAX_VALUE : 0);
		}
	}
	return result;
}

Image<bool> ZeroPadding(const Image<bool> & input) {
	auto result = Image<bool>(input.getHeight() + 2, input.getWidth() + 2, GRAY_CHANNELS, false);
	result.setPadding(PaddingType::Zero);
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto v = input.getValue(i, j, 0);
			result.setValue(i + 1, j + 1, 0, v);
		}
	}
	return result;
}

Image<bool> RemovePadding(const Image<bool> & input) {
	auto result = Image<bool>(input.getHeight() - 2, input.getWidth() - 2, GRAY_CHANNELS, false);
	result.setPadding(PaddingType::Zero);
	for (int i = 1; i < input.getHeight() - 1; i++) {
		for (int j = 1; j < input.getWidth() - 1; j++) {
			auto v = input.getValue(i, j, 0);
			result.setValue(i - 1, j - 1, 0, v);
		}
	}
	return result;
}

const int MP_HEIGHT = 3;

const int MP_WIDTH = 3;

const int MP_VERT_SHIFT = MP_HEIGHT / 2;

const int MP_HORI_SHIFT = MP_WIDTH / 2;

const bool CMP[][8][9] = {
	{
		/*1*/{0, 0, 1, /**/0, 1, 0, /**/0, 0, 0, },
		/*2*/{1, 0, 0, /**/0, 1, 0, /**/0, 0, 0, },
		/*3*/{0, 0, 0, /**/0, 1, 0, /**/1, 0, 0, },
		/*4*/{0, 0, 0, /**/0, 1, 0, /**/0, 0, 1, },
	},

	{
		/*1*/{0, 0, 0, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{0, 1, 0, /**/0, 1, 0, /**/0, 0, 0, },
		/*3*/{0, 0, 0, /**/1, 1, 0, /**/0, 0, 0, },
		/*4*/{0, 0, 0, /**/0, 1, 0, /**/0, 1, 0, },
	},

	{
		/*1*/{0, 0, 1, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{0, 1, 1, /**/0, 1, 0, /**/0, 0, 0, },
		/*3*/{1, 1, 0, /**/0, 1, 0, /**/0, 0, 0, },
		/*4*/{1, 0, 0, /**/1, 1, 0, /**/0, 0, 0, },
		/*5*/{0, 0, 0, /**/1, 1, 0, /**/1, 0, 0, },
		/*6*/{0, 0, 0, /**/0, 1, 0, /**/1, 1, 0, },
		/*7*/{0, 0, 0, /**/0, 1, 0, /**/0, 1, 1, },
		/*8*/{0, 0, 0, /**/0, 1, 1, /**/0, 0, 1, },
	},

	{
		/*1*/{0, 1, 0, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{0, 1, 0, /**/1, 1, 0, /**/0, 0, 0, },
		/*3*/{0, 0, 0, /**/1, 1, 0, /**/0, 1, 0, },
		/*4*/{0, 0, 0, /**/0, 1, 1, /**/0, 1, 0, },
	},

	{
		/*1*/{0, 0, 1, /**/0, 1, 1, /**/0, 0, 1, },
		/*2*/{1, 1, 1, /**/0, 1, 0, /**/0, 0, 0, },
		/*3*/{1, 0, 0, /**/1, 1, 0, /**/1, 0, 0, },
		/*4*/{0, 0, 0, /**/0, 1, 0, /**/1, 1, 1, },
	},

	{
		/*1*/{1, 1, 0, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{0, 1, 0, /**/0, 1, 1, /**/0, 0, 1, },
		/*3*/{0, 1, 1, /**/1, 1, 0, /**/0, 0, 0, },
		/*4*/{0, 0, 1, /**/0, 1, 1, /**/0, 1, 0, },
	},

	{
		/*1*/{0, 1, 1, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{1, 1, 0, /**/1, 1, 0, /**/0, 0, 0, },
		/*3*/{0, 0, 0, /**/1, 1, 0, /**/1, 1, 0, },
		/*4*/{0, 0, 0, /**/0, 1, 1, /**/0, 1, 1, },
	},

	{
		/*1*/{1, 1, 0, /**/0, 1, 1, /**/0, 0, 1, },
		/*2*/{0, 1, 1, /**/1, 1, 0, /**/1, 0, 0, },
	},

	{
		/*1*/{1, 1, 1, /**/0, 1, 1, /**/0, 0, 0, },
		/*2*/{0, 1, 1, /**/0, 1, 1, /**/0, 0, 1, },
		/*3*/{1, 1, 1, /**/1, 1, 0, /**/0, 0, 0, },
		/*4*/{1, 1, 0, /**/1, 1, 0, /**/1, 0, 0, },
		/*5*/{1, 0, 0, /**/1, 1, 0, /**/1, 1, 0, },
		/*6*/{0, 0, 0, /**/1, 1, 0, /**/1, 1, 1, },
		/*7*/{0, 0, 0, /**/0, 1, 1, /**/1, 1, 1, },
		/*8*/{0, 0, 1, /**/0, 1, 1, /**/0, 1, 1, },
	},

	{
		/*1*/{1, 1, 1, /**/0, 1, 1, /**/0, 0, 1, },
		/*2*/{1, 1, 1, /**/1, 1, 0, /**/1, 0, 0, },
		/*3*/{1, 0, 0, /**/1, 1, 0, /**/1, 1, 1, },
		/*4*/{0, 0, 1, /**/0, 1, 1, /**/1, 1, 1, },
	},

	{
		/*1*/{0, 1, 1, /**/0, 1, 1, /**/0, 1, 1, },
		/*2*/{1, 1, 1, /**/1, 1, 1, /**/0, 0, 0, },
		/*3*/{1, 1, 0, /**/1, 1, 0, /**/1, 1, 0, },
		/*4*/{0, 0, 0, /**/1, 1, 1, /**/1, 1, 1, },
	},

	{
		/*1*/{1, 1, 1, /**/0, 1, 1, /**/0, 1, 1, },
		/*2*/{0, 1, 1, /**/0, 1, 1, /**/1, 1, 1, },
		/*3*/{1, 1, 1, /**/1, 1, 1, /**/1, 0, 0, },
		/*4*/{1, 1, 1, /**/1, 1, 1, /**/0, 0, 1, },
		/*5*/{1, 1, 1, /**/1, 1, 0, /**/1, 1, 0, },
		/*6*/{1, 1, 0, /**/1, 1, 0, /**/1, 1, 1, },
		/*7*/{1, 0, 0, /**/1, 1, 1, /**/1, 1, 1, },
		/*8*/{0, 0, 1, /**/1, 1, 1, /**/1, 1, 1, },
	},

	{
		/*1*/{1, 1, 1, /**/0, 1, 1, /**/1, 1, 1, },
		/*2*/{1, 1, 1, /**/1, 1, 1, /**/1, 0, 1, },
		/*3*/{1, 1, 1, /**/1, 1, 0, /**/1, 1, 1, },
		/*4*/{1, 0, 1, /**/1, 1, 1, /**/1, 1, 1, },
	},

	{
		/*1*/{1, 1, 1, /**/1, 1, 1, /**/0, 1, 1, },
		/*2*/{1, 1, 1, /**/1, 1, 1, /**/1, 1, 0, },
		/*3*/{1, 1, 0, /**/1, 1, 1, /**/1, 1, 1, },
		/*4*/{0, 1, 1, /**/1, 1, 1, /**/1, 1, 1, },
	},
};

const vector<int> SHRINK_CMP = vector<int>{ 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, };

const vector<int> THIN_CMP = vector<int>{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, };

const vector<int> SKELETONIZE_CMP = vector<int>{ 3, 4, 8, 9, 10, 11, 12, 13, };

vector<Image<bool>> Patterns(const vector<int> & cmpIndices) {
	auto result = vector<Image<bool>>();
	for (auto i : cmpIndices) {
		for (auto & a : CMP[i]) {
			auto valid = false;
			for (int ii = 0; ii < MP_HEIGHT * MP_WIDTH; ii++) {
				if (a[ii]) {
					valid = true;
					break;
				}
			}
			if (valid) {
				result.push_back(Image<bool>(MP_HEIGHT, MP_WIDTH, GRAY_CHANNELS, a));
			}
		}
	}
	return result;
}

vector<Image<bool>> ShrinkConditionalMarkPatterns() {
	return Patterns(SHRINK_CMP);
}

vector<Image<bool>> ThinConditionalMarkPatterns() {
	return Patterns(THIN_CMP);
}

vector<Image<bool>> SkeletonizeConditionalMarkPatterns() {
	return Patterns(SKELETONIZE_CMP);
}

Image<bool> ApplyConditionalMarkPatterns(const Image<bool> & input, const vector<Image<bool>> & patterns) {
	auto result = Image<bool>(input.getHeight(), input.getWidth(), input.getChannel());
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto value = input.getValue(i, j, 0);
			//optimization
			if (!value) {
				result.setValue(i, j, 0, false);
				continue;
			}
			//process
			auto hit = false;
			for (auto & pattern : patterns) {
				for (int ii = 0; ii < MP_HEIGHT; ii++) {
					auto row = i + ii - MP_VERT_SHIFT;
					for (int jj = 0; jj < MP_WIDTH; jj++) {
						auto col = j + jj - MP_HORI_SHIFT;
						if (pattern.getValue(ii, jj, 0) != input.getValue(row, col, 0)) {
							goto NextPattern;
						}
					}
				}
				hit = true;
				break;
			NextPattern:
				hit = false;// necessary?
			}
			result.setValue(i, j, 0, hit);
		}
	}
	return result;
}

enum UnconditionalMarkType : int {
	Z = 0,
	M = 1,
	D,
	A,
	B,
	C,
};


const int UMP_ST[][9] = {
	//Spur
	{0, 0, M, /**/0, M, 0, /**/0, 0, 0, },
	{M, 0, 0, /**/0, M, 0, /**/0, 0, 0, },
	//Single 4-connection
	{0, 0, 0, /**/0, M, 0, /**/0, M, 0, },
	{0, 0, 0, /**/0, M, M, /**/0, 0, 0, },
	//L cluster
	{0, 0, M, /**/0, M, M, /**/0, 0, 0, },
	{0, M, M, /**/0, M, 0, /**/0, 0, 0, },
	{M, M, 0, /**/0, M, 0, /**/0, 0, 0, },
	{M, 0, 0, /**/M, M, 0, /**/0, 0, 0, },
	{0, 0, 0, /**/M, M, 0, /**/M, 0, 0, },
	{0, 0, 0, /**/0, M, 0, /**/M, M, 0, },
	{0, 0, 0, /**/0, M, 0, /**/0, M, M, },
	{0, 0, 0, /**/0, M, M, /**/0, 0, M, },
	//4-connected offset
	{0, M, M, /**/M, M, 0, /**/0, 0, 0, },
	{M, M, 0, /**/0, M, M, /**/0, 0, 0, },
	{0, M, 0, /**/0, M, M, /**/0, 0, M, },
	{0, 0, M, /**/0, M, M, /**/0, M, 0, },
	//spur corner cluster
	{0, A, M, /**/0, M, B, /**/M, 0, 0, },
	{M, B, 0, /**/A, M, 0, /**/0, 0, M, },
	{0, 0, M, /**/A, M, 0, /**/M, B, 0, },
	{M, 0, 0, /**/0, M, B, /**/0, A, M, },
	//Corner cluster
	{M, M, D, /**/M, M, D, /**/D, D, D, },
	//Tee branch
	{D, M, 0, /**/M, M, M, /**/D, 0, 0, },
	{0, M, D, /**/M, M, M, /**/0, 0, D, },
	{0, 0, D, /**/M, M, M, /**/0, M, D, },
	{D, 0, 0, /**/M, M, M, /**/D, M, 0, },
	{D, M, D, /**/M, M, 0, /**/0, M, 0, },
	{0, M, 0, /**/M, M, 0, /**/D, M, D, },
	{0, M, 0, /**/0, M, M, /**/D, M, D, },
	{D, M, D, /**/0, M, M, /**/0, M, 0, },
	//Vee branch
	{M, D, M, /**/D, M, D, /**/A, B, C, },
	{M, D, C, /**/D, M, B, /**/M, D, A, },
	{C, B, A, /**/D, M, D, /**/M, D, M, },
	{A, D, M, /**/B, M, D, /**/C, D, M, },
	//Diagonal branch
	{D, M, 0, /**/0, M, M, /**/M, 0, D, },
	{0, M, D, /**/M, M, 0, /**/D, 0, M, },
	{D, 0, M, /**/M, M, 0, /**/0, M, D, },
	{M, 0, D, /**/0, M, M, /**/D, M, 0, },
};

const int UMP_K[][9] = {
	//Spur
	{0, 0, 0, /**/0, M, 0, /**/0, 0, M, },
	{0, 0, 0, /**/0, M, 0, /**/M, 0, 0, },
	{0, 0, M, /**/0, M, 0, /**/0, 0, 0, },
	{M, 0, 0, /**/0, M, 0, /**/0, 0, 0, },
	//Single 4-connection
	{0, 0, 0, /**/0, M, 0, /**/0, M, 0, },
	{0, 0, 0, /**/0, M, M, /**/0, 0, 0, },
	{0, 0, 0, /**/M, M, 0, /**/0, 0, 0, },
	{0, M, 0, /**/0, M, 0, /**/0, 0, 0, },
	//L corner
	{0, M, 0, /**/0, M, M, /**/0, 0, 0, },
	{0, M, 0, /**/M, M, 0, /**/0, 0, 0, },
	{0, 0, 0, /**/0, M, M, /**/0, M, 0, },
	{0, 0, 0, /**/M, M, 0, /**/0, M, 0, },
	//Corner cluster
	{M, M, D, /**/M, M, D, /**/D, D, D, },
	{D, D, D, /**/D, M, M, /**/D, M, M, },
	//Tee branch
	{D, M, D, /**/M, M, M, /**/D, D, D, },
	{D, M, D, /**/M, M, D, /**/D, M, D, },
	{D, D, D, /**/M, M, M, /**/D, M, D, },
	{D, M, D, /**/D, M, M, /**/D, M, D, },
	//Vee branch
	{M, D, M, /**/D, M, D, /**/A, B, C, },
	{M, D, C, /**/D, M, B, /**/M, D, A, },
	{C, B, A, /**/D, M, D, /**/M, D, M, },
	{A, D, M, /**/B, M, D, /**/C, D, M, },
	//Diagonal branch
	{D, M, 0, /**/0, M, M, /**/M, 0, D, },
	{0, M, D, /**/M, M, 0, /**/D, 0, M, },
	{D, 0, M, /**/M, M, 0, /**/0, M, D, },
	{M, 0, D, /**/0, M, M, /**/D, M, 0, },
};

vector<Image<int>> ShrinkAndThinUnconditionalMarkPatterns() {
	auto result = vector<Image<int>>();
	for (auto & a : UMP_ST) {
		result.push_back(Image<int>(MP_HEIGHT, MP_WIDTH, GRAY_CHANNELS, a));
	}
	return result;
}

vector<Image<int>> SkeletonizeUnconditionalMarkPatterns() {
	auto result = vector<Image<int>>();
	for (auto & a : UMP_K) {
		result.push_back(Image<int>(MP_HEIGHT, MP_WIDTH, GRAY_CHANNELS, a));
	}
	return result;
}

Image<bool> ApplyUnconditionalMarkPatterns(const Image<bool> & input, const vector<Image<int>> & patterns) {
	auto result = Image<bool>(input.getHeight(), input.getWidth(), input.getChannel());
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto value = input.getValue(i, j, 0);
			//optimization
			if (!value) {
				result.setValue(i, j, 0, false);
				continue;
			}
			//process
			auto hit = false;			
			for (auto & pattern : patterns) {
				auto needCond = false;
				auto condPass = false;
				for (int ii = 0; ii < MP_HEIGHT; ii++) {
					auto row = i + ii - MP_VERT_SHIFT;
					for (int jj = 0; jj < MP_WIDTH; jj++) {
						auto col = j + jj - MP_HORI_SHIFT;
						auto m = pattern.getValue(ii, jj, 0);
						auto v = input.getValue(row, col, 0);
						switch (m) {
							case Z:
								if (v != false) {
									goto NextPattern;
								}
								break;
							case M:
								if (v != true) {
									goto NextPattern;
								}
								break;
							case A:
							case B:
							case C:
								needCond = true;
								if (v == true) {
									condPass = true;
								}
								break;
						}
					}
				}
				if (!needCond || condPass) {
					hit = true;
				}
				break;
			NextPattern:
				hit = false;// necessary?
			}
			result.setValue(i, j, 0, hit);
		}
	}
	return result;
}

Image<bool> Bridging(const Image<bool> & input) {
	auto result = Image<bool>(input.getHeight(), input.getWidth(), input.getChannel());
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto x = input.getValue(i, j, 0);
			/*bool xs[8] = {
				input.getValue(i, j + 1, 0), 
				input.getValue(i - 1, j + 1, 0), 
				input.getValue(i - 1, j, 0),
				input.getValue(i - 1, j - 1, 0),
				input.getValue(i, j - 1, 0),
				input.getValue(i + 1, j - 1, 0),
				input.getValue(i + 1, j, 0),
				input.getValue(i + 1, j + 1, 0),
			};*/
			bool xs[8];
			xs[0] = input.getValue(i, j + 1, 0);
			xs[1] = input.getValue(i - 1, j + 1, 0);
			xs[2] = input.getValue(i - 1, j, 0);
			xs[3] = input.getValue(i - 1, j - 1, 0);
			xs[4] = input.getValue(i, j - 1, 0);
			xs[5] = input.getValue(i + 1, j - 1, 0);
			xs[6] = input.getValue(i + 1, j, 0);
			xs[7] = input.getValue(i + 1, j + 1, 0);

			auto L1 = !x && !xs[0] && xs[1] && !xs[2] && xs[3] && !xs[4] && !xs[5] && !xs[6] && !xs[7];
			auto L2 = !x && !xs[0] && !xs[1] && !xs[2] && xs[3] && !xs[4] && xs[5] && !xs[6] && !xs[7];
			auto L3 = !x && !xs[0] && !xs[1] && !xs[2] && !xs[3] && !xs[4] && xs[5] && !xs[6] && xs[7];
			auto L4 = !x && !xs[0] && xs[1] && !xs[2] && !xs[3] && !xs[4] && !xs[5] && !xs[6] && xs[7];
			auto PQ = L1 || L2 || L3 || L4;
			auto P1 = !xs[2] && !xs[6] && (xs[3] || xs[4] || xs[5]) && (xs[0] || xs[1] || xs[7]) && !PQ;
			auto P2 = !xs[0] && !xs[4] && (xs[1] || xs[2] || xs[3]) && (xs[5] || xs[6] || xs[7]) && !PQ;
			auto P3 = !xs[0] && !xs[6] && xs[7] && (xs[2] || xs[3] || xs[4]);
			auto P4 = !xs[0] && !xs[2] && xs[1] && (xs[4] || xs[5] || xs[6]);
			auto P5 = !xs[2] && !xs[4] && xs[3] && (xs[0] || xs[6] || xs[7]);
			auto P6 = !xs[4] && !xs[6] && xs[5] && (xs[0] || xs[1] || xs[2]);
			auto pixel = x || (P1 || P2 || P3 || P4 || P5 || P6);
			result.setValue(i, j, 0, pixel);
		}
	}
	return result;
}

enum class MorphologicalOperation {
	Shrink,
	Thin,
	Skeletonize,
};

Image<bool> MorphologicalProcessing(const Image<bool> & input, const vector<Image<bool>> & cmps, const vector<Image<int>> & umps, const bool extendBoundary, const bool isSkeletonizing = false) {
	auto temp = ZeroPadding(input);
	auto iter = 0;
	auto changed = true;	
	while (changed) {
		iter++;
		changed = false;
		auto M = ApplyConditionalMarkPatterns(temp, cmps);
		auto P = ApplyUnconditionalMarkPatterns(M, umps);
		auto result = Image<bool>(temp.getHeight(), temp.getWidth(), temp.getChannel());
		if(iter==0){
			Strech(M).writeToFile("tmp.raw");
			
		}
		for (int i = 0; i < temp.getHeight(); i++) {
			for (int j = 0; j < temp.getWidth(); j++) {
				auto x = temp.getValue(i, j, 0);
				auto m = M.getValue(i, j, 0);
				auto p = P.getValue(i, j, 0);
				auto pixel = x && (!m || p);
				result.setValue(i, j, 0, pixel);
				if (pixel != x) {
					changed = true;
				}
			}
		}
		temp = result;
	}	
	
	if (isSkeletonizing) {
		temp = Bridging(temp);
	}
	temp = RemovePadding(temp);
	return temp;
}

Image<unsigned char> MorphologicalProcessing(const Image<unsigned char> & input, const vector<Image<bool>> & cmps, const vector<Image<int>> & umps, const bool extendBoundary, const bool isSkeletonizing = false) {
	auto temp = Binarize(input);	
	return Strech(MorphologicalProcessing(temp, cmps, umps, extendBoundary, isSkeletonizing));
}

Image<bool> Shrink(const Image<bool> & input) {
	auto cmps = ShrinkConditionalMarkPatterns();
	auto umps = ShrinkAndThinUnconditionalMarkPatterns();
	return MorphologicalProcessing(input, cmps, umps, true);
}

Image<unsigned char> Shrink(const Image<unsigned char> & input) {
	auto cmps = ShrinkConditionalMarkPatterns();
	auto umps = ShrinkAndThinUnconditionalMarkPatterns();
	return MorphologicalProcessing(input, cmps, umps, true);
}

Image<unsigned char> Thin(const Image<unsigned char> & input) {
	auto cmps = ThinConditionalMarkPatterns();
	auto umps = ShrinkAndThinUnconditionalMarkPatterns();
	return MorphologicalProcessing(input, cmps, umps, true);
}

Image<unsigned char> Skeletonize(const Image<unsigned char> & input) {
	auto cmps = SkeletonizeConditionalMarkPatterns();
	auto umps = SkeletonizeUnconditionalMarkPatterns();
	return MorphologicalProcessing(input, cmps, umps, true, true);
}

/*=================================
|                                 |
|                b)               |
|                                 |
=================================*/

int FloodFill(Image<bool> & shrink, const bool fillColor, const int row, const int col) {
	auto result = 0;
	auto s = stack<tuple<int, int>>();
	s.push(make_tuple(row, col));
	while (!s.empty()) {
		auto coordinate = s.top();
		s.pop();
		auto r = get<0>(coordinate);
		auto c = get<1>(coordinate);
		if (r < 0 || c < 0 || r >= shrink.getHeight() || c >= shrink.getWidth()) {
			continue;
		}
		if (shrink.getValue(r, c, 0) == fillColor) {
			continue;
		}
		shrink.setValue(r, c, 0, fillColor);
		result++;
		s.push(make_tuple(r - 1, c));
		s.push(make_tuple(r + 1, c));
		s.push(make_tuple(r, c - 1));
		s.push(make_tuple(r, c + 1));
	}
	return result;
}

struct Ring {
	int Row;
	int Col;
	int Area;
};

vector<Ring> CountRings(const Image<bool> & input) {
	auto temp = input;
	// Fill Background
	FloodFill(temp, true, 0, 0);
	// Count
	auto result = vector<Ring>();
	auto changed = true;
	while (changed) {
		changed = false;
		for (int i = 0; i < temp.getHeight(); i++) {
			for (int j = 0; j < temp.getWidth(); j++) {
				if (!temp.getValue(i, j, 0)) {
					auto ring = Ring();
					ring.Row = i;
					ring.Col = j;
					ring.Area = FloodFill(temp, true, i, j);
					result.push_back(ring);
					changed = true;
				}
			}
		}
	}
	return result;
}

int CorrectDefect(Image<bool> & shrink, Image<bool> & correction, const int row, const int col) { // modified flood-fill
	auto result = 0;
	auto s = stack<tuple<int, int>>();
	s.push(make_tuple(row, col));
	while (!s.empty()) {
		auto coordinate = s.top();
		s.pop();
		auto r = get<0>(coordinate);
		auto c = get<1>(coordinate);
		if (r < 0 || c < 0 || r >= shrink.getHeight() || c >= shrink.getWidth()) {
			continue;
		}
		if (shrink.getValue(r, c, 0)) {
			continue;
		}
		shrink.setValue(r, c, 0, true);
		if (!correction.getValue(r, c, 0)) { // defect point
			result += FloodFill(correction, true, r, c);
		}
		s.push(make_tuple(r - 1, c));
		s.push(make_tuple(r + 1, c));
		s.push(make_tuple(r, c - 1));
		s.push(make_tuple(r, c + 1));
	}
	return result;
}

tuple<int, Image<bool>> CorrectDefects(const Image<bool> & input, const int defectAreaThreshold) {
	auto image = input;
	auto shrink = Shrink(image);
	auto rings = CountRings(shrink);
	auto defectCount = 0;
	for (auto & r : rings) {
		auto temp = image;
		auto defectArea = CorrectDefect(shrink, temp, r.Row, r.Col);
		if (defectArea < defectAreaThreshold) {
			image = temp;
			defectCount++;
		}
	}
	return make_tuple(defectCount, image);
}

tuple<int, Image<unsigned char>> CorrectDefects(const Image<unsigned char> & input, const int defectAreaThreshold) {
	auto result = CorrectDefects(Binarize(input), defectAreaThreshold);
	return make_tuple(get<0>(result), Strech(get<1>(result)));
}

/*=================================
|                                 |
|                c)               |
|                                 |
=================================*/

const int RICE_TYPES = 11;

struct RGB {
	unsigned char R;
	unsigned char G;
	unsigned char B;

	RGB() : R(0), G(0), B(0) {

	}

	RGB(const unsigned char r, const unsigned char g, const unsigned char b) : R(r), G(g), B(b) {

	}

	double Distance(const RGB & other) const {
		return sqrt(pow(R - other.R, 2) + pow(G - other.G, 2) + pow(B - other.B, 2));
	}

	unsigned char ToGray() {
		return round((R + B + G) / 3);
	}
};

Image<bool> BackgroundObjectBinarize(const Image<unsigned char> & input, const RGB & color, const double distanceThreshold) {
	if (input.getChannel() != COLOR_CHANNELS) {
		throw ArgumentException;
	}
	auto result = Image<bool>(input.getHeight(), input.getWidth(), GRAY_CHANNELS);
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			auto dist = color.Distance(RGB(input.getValue(i, j, 0), input.getValue(i, j, 1), input.getValue(i, j, 2)));
			result.setValue(i, j, 0, dist >= distanceThreshold);
		}
	}
	//Strech(result).writeToFile("binarized.raw");
	return result;
}

struct Coordinate {
	double Row;
	double Col;
};

struct Rice {
	int Row;
	int Col;
	int Area;

	Rice(const int row, const int col, const int area) : Row(row), Col(col), Area(area) {}

	double Distance(const Rice & other) const {
		return sqrt(pow(Row - other.Row, 2) + pow(Col - other.Col, 2));
	}

	double Distance(const Coordinate & cluster) const {
		return sqrt(pow(Row - cluster.Row, 2) + pow(Col - cluster.Col, 2));
	}
};

vector<Rice> CountRice(const Image<unsigned char> & input, const double distanceThreshold, const int defectAreaThreashold) {
	if (input.getChannel() != COLOR_CHANNELS) {
		throw ArgumentException;
	}
	// binarize
	auto temp = BackgroundObjectBinarize(input, RGB(input.getValue(0, 0, 0), input.getValue(0, 0, 1), input.getValue(0, 0, 2)), distanceThreshold);
	// remove spots out of rices
	auto riceImage = temp;
	for (int i = 0; i < input.getHeight(); i++) {
		for (int j = 0; j < input.getWidth(); j++) {
			if (temp.getValue(i, j, 0)) {
				auto area = FloodFill(temp, false, i, j);
				if (area >= defectAreaThreashold) {					
					//keep
				} else {
					FloodFill(riceImage, false, i, j);// remove no object regions, only leave rice regions
				}
			}			
		}
	}
	// remove holes inside rices
	auto corrected = get<1>(CorrectDefects(riceImage, numeric_limits<int>::max()));
	//Strech(corrected).writeToFile("corrected.raw");
	// calculate rice centers
	auto shrink = Shrink(corrected);
	//Strech(shrink).writeToFile("shrinked.raw");
	//calculate area of each rice
	auto rices = vector<Rice>();
	for (int i = 0; i < shrink.getHeight(); i++) {
		for (int j = 0; j < shrink.getWidth(); j++) {
			if (shrink.getValue(i, j, 0)) {
				auto area = FloodFill(riceImage, false, i, j);
				if (area > 0) { //TODO: something goes wrong? why shrinking doesn't work this time?
					rices.push_back(Rice(i, j, area));
				} else {
					//cerr << "Wrong result" << endl << endl;
				}
			}
		}
	}
	sort(rices.begin(), rices.end(), [](Rice & a, Rice & b) -> bool { return a.Area < b.Area; });
	return rices;
}

int IndexOfCluster(const Rice & rice, const vector<Coordinate> clusterCenters) {
	auto result = 0;
	double minDist = rice.Distance(clusterCenters[result]);
	for (int i = 1; i < clusterCenters.size(); i++) {
		auto dist = rice.Distance(clusterCenters[i]);
		if (dist < minDist) {
			minDist = dist;
			result = i;
		}
	}
	return result;
}

double CalcError(const vector<Coordinate> & clusterCenters, const vector<vector<Rice>> & clusters) {
	if (clusterCenters.size() != clusters.size()) {
		throw ArgumentException;
	}
	auto result = 0.0;
	for (int i = 0; i < clusterCenters.size(); i++) {
		auto cluster = clusters[i];
		for (int j = 0; j < cluster.size(); j++) {
			result += cluster[j].Distance(clusterCenters[i]);
		}
	}
	return result;
}

Coordinate CalcCenter(const vector<Rice> & cluster) {
	auto result = Coordinate();
	for (auto & rice : cluster) {
		result.Row += double(rice.Row) / cluster.size();
		result.Col += double(rice.Col) / cluster.size();;
	}
	return result;
}

vector<vector<Rice>> Kmeans(const vector<Rice> & rices, const int k, const int trailThreshold) {
	auto result = vector<vector<Rice>>(k);
	auto resultCenters = vector<Coordinate>(k);
	auto resultError = numeric_limits<double>::max();
	auto resultTrial = -1;
	srand(0);
	auto trial = 0;
	do {
		auto clusterCenters = vector<Coordinate>(k);
		auto clusters = vector<vector<Rice>>(k);
		trial++;
		//init
		auto temp = rices;
		for (int i = 0; i < k; i++) {
			auto index = rand() % temp.size();
			auto rice = temp[index];//[i * (rices.size() / k)];	
			temp.erase(temp.begin() + index);
			clusterCenters[i].Row = rice.Row;
			clusterCenters[i].Col = rice.Col;
		}
		auto iter = 0;
		auto lastError = 0.0;
		while (true) {
			iter++;
			// clear clusters
			for (int i = 0; i < k; i++) {
				clusters[i].clear();
			}
			// calc clusters
			for (auto & rice : rices) {
				clusters[IndexOfCluster(rice, clusterCenters)].push_back(rice);
			}
			// update centers
			for (int i = 0; i < k; i++) {
				clusterCenters[i] = CalcCenter(clusters[i]);
			}
			// calc error
			auto newError = CalcError(clusterCenters, clusters);
			if (newError == lastError) {
				break;
			}
			lastError = newError;
		}

		//check
		if (lastError < resultError) {
			resultError = lastError;
			result = clusters;
			resultCenters = clusterCenters;
			resultTrial = trial;
		}
	} while (trial < trailThreshold);
	return result;
}

double CalcAverageArea(const vector<Rice> & cluster) {
	if (cluster.size() == 0) {
		throw InvalidOperationException;
	}
	auto result = 0.0;
	for (auto & rice : cluster) {
		result += rice.Area;
	}
	result /= cluster.size();
	return result;
}

vector<vector<Rice>> SortClusters(const vector<vector<Rice>> & clusters) {
	auto result = clusters;
	//sort rices within each cluster
	for (auto & cluster : result) {
		sort(cluster.begin(), cluster.end(), [](Rice & a, Rice & b) -> bool { return a.Area < b.Area; });
	}
	//sort cluster by average area
	sort(result.begin(), result.end(), [](const vector<Rice> & a, const vector<Rice> & b) -> bool { return CalcAverageArea(a) < CalcAverageArea(b); });	
	return result;
}

vector<vector<Rice>> SortRices(const vector<Rice> & rices, const int k, const int trailThreshold) {
	return SortClusters(Kmeans(rices, k, trailThreshold));
}

void DisplayRiceResult(const vector<vector<Rice>> & result) {
	cout << "All rice grains are classified automatically. Sorted results are shown below." << endl;
	cout << "Type\tRow\tCol\tArea" << endl;
	for (int i = 0; i < result.size(); i++) {
		cout << endl;
		for (auto & rice : result[i]) {
			cout << i + 1 << "\t" << rice.Row << "\t" << rice.Col << "\t" << rice.Area << endl;
		}
	}
}

/*=================================
|                                 |
|              main               |
|                                 |
=================================*/

const char * OPTION_FUNCTION = "-f";
const char * OPTION_DEFECT_AREA_THRESHOLD = "-a";
const char * OPTION_COLOR_DISTANCE_THRESHOLD = "-d";
const char * OPTION_KMEANS_K = "-k";
const char * OPTION_KMEANS_TRIAL_THRESHOLD = "-t";
const char * OPTION_OUTPUT = "-o";
const char * OPTION_HEIGHT = "-h";
const char * OPTION_WIDTH = "-w";
const char * OPTION_CHANNEL = "-c";

const char * FUNCTION_SHRINK = "s";
const char * FUNCTION_THIN = "t";
const char * FUNCTION_SKELETONIZE = "k";
const char * FUNCTION_DEER = "d";
const char * FUNCTION_RICE = "r";

enum class FunstionType {
	Shrink,
	Thin,
	Skeletonize,
	Deer,
	Rice,
};

const char * WrongCommandException = "Wrong command";

FunstionType ParseFunction(const string & cmd) {
	if (cmd == FUNCTION_SHRINK) {
		return FunstionType::Shrink;
	} else if (cmd == FUNCTION_THIN) {
		return FunstionType::Thin;
	} else if (cmd == FUNCTION_SKELETONIZE) {
		return FunstionType::Skeletonize;
	} else if (cmd == FUNCTION_DEER) {
		return FunstionType::Deer;
	} else if (cmd == FUNCTION_RICE) {
		return FunstionType::Rice;
	} else {
		throw WrongCommandException;
	}
}

const FunstionType DEFAULT_FUNCTION = FunstionType::Shrink;
const int DEFAULT_CHANNEL = GRAY_CHANNELS;
const int DEFAULT_KMEANS_K = RICE_TYPES;
const int DEFAULT_DEFECT_AREA_THRESHOLD = 10;
const int DEFAULT_COLOR_DISTANCE_THRESHOLD = 16;
const int DEFAULT_KMEANS_TRAIL_THRESHOLD = 100;

void PrintUsage() {
	cerr << "Usage:" << endl
		<< "\t" << "Problem2 [OPTION]... [INPUT FILE]" << endl
		<< endl
		<< "Intro:" << endl
		<< "\t" << "Morphological Processing." << endl
		<< "\t" << "For USC EE569 2019 spring home work 3 problem 2 by Zongjian Li." << endl
		<< endl
		<< "Options:" << endl
		<< "\t" << OPTION_FUNCTION << "\t" << "Functions." << endl
		<< "\t\t" << "You can choose from \"" << FUNCTION_SHRINK << "\"(Shrink), \"" << FUNCTION_THIN << "\"(Thin), \"" << FUNCTION_SKELETONIZE << "\"(Skeletonize), \"" << FUNCTION_DEER << "\"(Deer, problem b), \"" << FUNCTION_RICE << "\"(Rice, problem c)." << endl
		<< "\t\t" << "The default is Shrink." << endl		
		<< "\t" << OPTION_DEFECT_AREA_THRESHOLD << "\t" << "Defect area threshold. Connected region with area smaller than this threshold will be consider as defects. Both problem b and c use this parameter. The default is " << DEFAULT_DEFECT_AREA_THRESHOLD << "." << endl
		<< "\t" << OPTION_COLOR_DISTANCE_THRESHOLD << "\t" << "Threshold of color distance. Used in problem c to determine whether a pixel is the background. The default is " << DEFAULT_COLOR_DISTANCE_THRESHOLD << "." << endl
		<< "\t" << OPTION_KMEANS_K << "\t" << "Number of types of rice. K in K-means algorithm. Used in problem c. The default is " << DEFAULT_KMEANS_K << "." << endl
		<< "\t" << OPTION_KMEANS_TRIAL_THRESHOLD << "\t" << "Number of trials of K-means. The best result will be taken among trials. Used in problem c. The default is " << DEFAULT_KMEANS_TRAIL_THRESHOLD << "." << endl
		<< "\t" << OPTION_OUTPUT << "\t" << "Output filename. The default is \"" << DEFAULT_OUTPUT_FILENAME << "\"." << endl
		<< "\t" << OPTION_HEIGHT << "\t" << "Height of the input image. The default is " << DEFAULT_HEIGHT << "." << endl
		<< "\t" << OPTION_WIDTH << "\t" << "Width of the input image. The default is " << DEFAULT_WIDTH << "." << endl
		<< "\t" << OPTION_CHANNEL << "\t" << "Number of channels of the input image. The default is " << GRAY_CHANNELS << "." << endl
		<< endl
		<< "Example:" << endl
		<< "\t" << "Problem2 -f " << FUNCTION_SHRINK << " -o my_output_image.raw my_input_image.raw" << endl
		<< endl;
}

int main(int argc, char *argv[]) {
	auto function = DEFAULT_FUNCTION;
	auto defectAreaThreshold = DEFAULT_DEFECT_AREA_THRESHOLD;
	auto colorDistanceThreshold = DEFAULT_COLOR_DISTANCE_THRESHOLD;
	auto k = DEFAULT_KMEANS_K;
	auto trialThreshold = DEFAULT_KMEANS_TRAIL_THRESHOLD;
	auto output = string(DEFAULT_OUTPUT_FILENAME);
	auto height = DEFAULT_HEIGHT;
	auto width = DEFAULT_WIDTH;
	auto channel = GRAY_CHANNELS;

	auto functionFlag = false;
	auto defectAreaThresholdFlag = false;
	auto colorDistanceThresholdFlag = false;
	auto kFlag = false;
	auto trialThresholdFlag = false;
	auto outputFlag = false;
	auto heightFlag = false;
	auto widthFlag = false;
	auto channelFlag = false;
	auto input = string();

#if defined(DEBUG) || defined(_DEBUG)
	cerr << "WARNNING: You are running this program under DEBUG mode which is extremely SLOW! RELEASE mode will give you several handurd speed up in this problem." << endl << endl;
#endif 

	try {
		int i;
		for (i = 1; i < argc; i++) {
			auto cmd = string(argv[i]);
			if (functionFlag) {
				function = ParseFunction(cmd);
				functionFlag = false;
			} else if (defectAreaThresholdFlag) {
				defectAreaThreshold = atoi(cmd.c_str());
				defectAreaThresholdFlag = false;
			} else if (colorDistanceThresholdFlag) {
				colorDistanceThreshold = atoi(cmd.c_str());
				colorDistanceThresholdFlag = false;
			} else if (kFlag) {
				k = atoi(cmd.c_str());
				kFlag = false;
			} else if (trialThresholdFlag) {
				trialThreshold = atoi(cmd.c_str());
				trialThresholdFlag = false;
			} else if (outputFlag) {
				output = cmd;
				outputFlag = false;
			} else if (heightFlag) {
				height = atoi(cmd.c_str());
				heightFlag = false;
			} else if (widthFlag) {
				width = atoi(cmd.c_str());
				widthFlag = false;
			} else if (channelFlag) {
				channel = atoi(cmd.c_str());
				channelFlag = false;
			} else if (cmd == OPTION_FUNCTION) {
				functionFlag = true;
			} else if (cmd == OPTION_DEFECT_AREA_THRESHOLD) {
				defectAreaThresholdFlag = true;
			} else if (cmd == OPTION_COLOR_DISTANCE_THRESHOLD) {
				colorDistanceThresholdFlag = true;
			} else if (cmd == OPTION_KMEANS_K) {
				kFlag = true;
			} else if (cmd == OPTION_KMEANS_TRIAL_THRESHOLD) {
				trialThresholdFlag = true;
			} else if (cmd == OPTION_OUTPUT) {
				outputFlag = true;
			} else if (cmd == OPTION_HEIGHT) {
				heightFlag = true;
			} else if (cmd == OPTION_WIDTH) {
				widthFlag = true;
			} else if (cmd == OPTION_CHANNEL) {
				channelFlag = true;
			} else {
				input = cmd;
				break;
			}
		}
		if (input == "" || i != argc - 1 || functionFlag || defectAreaThresholdFlag || colorDistanceThresholdFlag || kFlag || trialThresholdFlag || outputFlag || heightFlag || widthFlag || channelFlag) {
			PrintUsage();
			throw WrongCommandException;
		}
		cout << input << endl;
		auto in = Image<unsigned char>(height, width, channel, input);
		Image<unsigned char> out;
		tuple<int, Image<unsigned char>> deerResult;
		vector<Rice> rices;

		switch (function) {
			case FunstionType::Shrink:
				out = Shrink(in);
				break;
			case FunstionType::Thin:
				out = Thin(in);
				break;
			case FunstionType::Skeletonize:
				out = Skeletonize(in);
				break;
			case FunstionType::Deer:
				deerResult = CorrectDefects(in, defectAreaThreshold);
				cout << "# of defects is " << get<0>(deerResult) << " (parameters: defect_area_threshold = " << defectAreaThreshold << ")" << endl;
				cout << "NOTE: Corrected image is stored as \"" << output << "\"." << endl;
				out = get<1>(deerResult);
				break;
			case FunstionType::Rice:
				rices = CountRice(in, colorDistanceThreshold, defectAreaThreshold);
				cout << "# of rice grains is " << rices.size() << " (parameters: color_distance_threshold = " << colorDistanceThreshold << ", defect_area_threshold = " << defectAreaThreshold << ", K = " << k << ", #_of_trial = " << trialThreshold << ")" << endl;
				DisplayRiceResult(SortRices(rices, RICE_TYPES, trialThreshold));
				break;
			default:
				throw InvalidOperationException;
		}
		
		if (out.getHeight() * out.getWidth() > 0) {
			out.writeToFile(output);
		}
		return 0;
	} catch (const char * ex) {
		cerr << "Captured exception: " << ex << endl;
	}
	return 1;	
}