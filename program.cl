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
__kernel void device_function( write_only image2d_t pattern, float t )
#else
__kernel void device_function( __global uint* pattern, __global uint* second, uint pw, uint ph)
#endif
{
	// adapted from inigo quilez - iq/2013
	uint idx = get_global_id( 0 );
	uint idy = get_global_id( 1 );
	uint id = idx + 512 * idy;
	if (id >= (512 * 512)) return;
	float3 col = (float3)( 20.f, 0.f, 0.f );
	
	// count active neighbors
	uint n = GetBit( second, pw, idx - 1, idy - 1 ) + GetBit( second, pw, idx, idy - 1 ) + GetBit( second, pw, idx + 1, idy - 1 ) + GetBit( second, pw, idx - 1, idy ) + 
			 GetBit( second, pw, idx + 1, idy ) + GetBit( second, pw, idx - 1, idy + 1 ) + GetBit( second, pw, idx, idy + 1 ) + GetBit( second, pw, idx + 1, idy + 1 );
	if ((GetBit( second, pw, idx, idy ) == 1 && n == 2) || n == 3) { BitSet( pattern, pw, idx, idy ); }
	
#ifdef GLINTEROP
	int2 pos = (int2)(idx,idy);
	write_imagef( pattern, pos, (float4)(col * (1.0f / 16.0f), 1.0f ) );
#else
	int r = (int)clamp( 16.0f * col.x, 0.f, 255.f );
	int g = (int)clamp( 16.0f * col.y, 0.f, 255.f );
	int b = (int)clamp( 16.0f * col.z, 0.f, 255.f );
	pattern[id] = (r << 16) + (g << 8) + b;
#endif
}