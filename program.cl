#define GLINTEROP

// function for seeing if there's a bit alive at a given position in a given array
uint GetBit(__global uint* second, uint pw, uint x, uint y) {
	uint i = pw * y + (x >> 5);
    return (second[i] >> (int)(x & 31)) & 1U;
}

// function for setting an alive bit at a given position in a given array
void BitSet(__global uint* pattern, uint pw, uint x, uint y) {
	uint i = pw * y + (x >> 5);
	atomic_or(&pattern[i], 1U << (int)(x & 31));
}

#ifdef GLINTEROP
__kernel void device_function( write_only image2d_t image, __global uint* pattern, __global uint* second, uint pw, uint ph, int sWidth, int sHeight, uint xoffset, uint yoffset )
#else
__kernel void device_function( __global uint* pattern, __global uint* second, __global uint* teken, uint pw, uint ph, int sWidth, int sHeight, uint xoffset, uint yoffset )
#endif
{
	uint idx = get_global_id( 0 );
	uint idy = get_global_id( 1 );
	if (idx > 1 && idx <= pw * 32 - 1 && idy > 1 && idy <= ph) { //only execute if the current bit is within reasonable bounds
		// set bit to 0
		atomic_and(&pattern[pw * idy + (idx >> 5)], ~(1U << (idx & 31)));
		
		// count active neighbors
		uint n = GetBit( second, pw, idx - 1, idy - 1 ) + GetBit( second, pw, idx, idy - 1 ) + GetBit( second, pw, idx + 1, idy - 1 ) + GetBit( second, pw, idx - 1, idy ) + 
				 GetBit( second, pw, idx + 1, idy ) + GetBit( second, pw, idx - 1, idy + 1 ) + GetBit( second, pw, idx, idy + 1 ) + GetBit( second, pw, idx + 1, idy + 1 );
		// set the bit to alive if it should be
		if ((GetBit( second, pw, idx, idy ) == 1 && n == 2) || n == 3) { BitSet( pattern, pw, idx, idy ); }
		
		// stop if the bit doesn't need to be drawn
		if (idx < xoffset || idx >= sWidth + xoffset || idy < yoffset || idy >= sHeight + yoffset) return;
		
		// define the colour, depending of whether or not the bit is alive
		float3 col;
		if (GetBit( second, pw, idx, idy )){
			col = (float3)( 16.f, 16.f, 16.f );
		}
		else {
			col = (float3)( 0.f, 0.f, 0.f );
		}
	
#ifdef GLINTEROP
		// calculate position, enhanced after mouse dragging
		int2 pos = (int2)(idx - xoffset, idy - yoffset);
		// write directly to the texture
		write_imagef( image, pos, (float4)(col * (1.0f / 16.0f), 1.0f ) );
#else
		// scale and clamp colour
		int r = (int)clamp( 16.0f * col.x, 0.f, 255.f );
		int g = (int)clamp( 16.0f * col.y, 0.f, 255.f );
		int b = (int)clamp( 16.0f * col.z, 0.f, 255.f );
		
		// calculate position, enhanced after mouse dragging
		idx -= xoffset;
		idy -= yoffset;
		
		// calculate id in array
		uint id = idx + sWidth * idy;
		
		// set the right color for the current id
		teken[id] = (r << 16) + (g << 8) + b;
#endif
	}
}

// kernel for swapping the current bit in the two arrays
__kernel void ruiltransactie( __global uint* pattern, __global uint* second, uint pw, uint ph ) {
	uint idx = get_global_id( 0 );
	uint idy = get_global_id( 1 );
	// see if the bit was alive in the round we just calculated
	uint oldVal = GetBit(pattern, pw, idx, idy) << (idx & 31);
	uint i = pw * idy + (idx >> 5);
	// set the bit in the array to be overwritten to 0
	atomic_and(&second[i], ~(1U << (idx & 31)));
	// then set it to the value we just checked
	atomic_or(&second[i], oldVal);
}