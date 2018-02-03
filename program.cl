uint GetBit(__global uint* second, uint pw, uint x, uint y) {
	uint i = pw * y + (x >> 5);
    return (second[i] >> (int)(x & 31)) & 1U;
}

void BitSet(__global uint* pattern, uint pw, uint x, uint y) {
	uint i = pw * y + (x >> 5);
	atomic_or(&pattern[i], 1U << (int)(x & 31));
}

// #define GLINTEROP

#ifdef GLINTEROP
__kernel void device_function( write_only image2d_t pattern, write_only image2d_t second, uint pw )
#else
__kernel void device_function( __global uint* pattern, __global uint* second, __global uint* teken, uint pw, uint ph, int sWidth, int sHeight, uint xoffset, uint yoffset )
#endif
{
	// adapted from inigo quilez - iq/2013
	uint idx = get_global_id( 0 );
	uint idy = get_global_id( 1 );
	atomic_and(&pattern[pw * idy + (idx >> 5)], ~(1U << (idx & 31)));
	
	// count active neighbors
	if (idx >= 1 && idx <= pw * 32 - 1 && idy >= 1 && idy <= ph) {
		uint n = GetBit( second, pw, idx - 1, idy - 1 ) + GetBit( second, pw, idx, idy - 1 ) + GetBit( second, pw, idx + 1, idy - 1 ) + GetBit( second, pw, idx - 1, idy ) + 
				 GetBit( second, pw, idx + 1, idy ) + GetBit( second, pw, idx - 1, idy + 1 ) + GetBit( second, pw, idx, idy + 1 ) + GetBit( second, pw, idx + 1, idy + 1 );
		if ((GetBit( second, pw, idx, idy ) == 1 && n == 2) || n == 3) { BitSet( pattern, pw, idx, idy ); }
	}
	
	if (idx < xoffset || idx >= sWidth + xoffset || idy < yoffset || idy >= sHeight + yoffset) return;
	float3 col;
	if (GetBit( second, pw, idx, idy )){
		col = (float3)( 16.f, 16.f, 16.f );
	}
	else {
		col = (float3)( 0.f, 0.f, 0.f );
	}
	
#ifdef GLINTEROP
	int2 pos = (int2)(idx,idy);
	write_imagef( pattern, pos, (float4)(col * (1.0f / 16.0f), 1.0f ) );
#else
	int r = (int)clamp( 16.0f * col.x, 0.f, 255.f );
	int g = (int)clamp( 16.0f * col.y, 0.f, 255.f );
	int b = (int)clamp( 16.0f * col.z, 0.f, 255.f );
	
	idx -= xoffset;
	idy -= yoffset;
	uint id = idx + sWidth * idy;
	
	teken[id] = (r << 16) + (g << 8) + b;
#endif
}

__kernel void ruiltransactie( __global uint* pattern, __global uint* second, uint pw, uint ph, int sWidth, int sHeight, uint xoffset, uint yoffset ) {
	uint idx = get_global_id( 0 );// + xoffset;
	uint idy = get_global_id( 1 );// + yoffset;
	uint oldVal = GetBit(pattern, pw, idx, idy) << (idx & 31);
	uint i = pw * idy + (idx >> 5);
	atomic_and(&second[i], ~(1U << (idx & 31)));
	atomic_or(&second[i], oldVal);
}