#include "Math.hpp"
#include "ProcessAlpha.hpp"
#include "ProcessCommon.hpp"
#include "Tables.hpp"
#include "Types.hpp"
#include "Vector.hpp"

static uint Average1( const uint8* data )
{
    uint32 a = 4;
    for( int i=0; i<8; i++ )
    {
        a += *data++;
    }
    return a / 8;
}

static void CalcErrorBlock( const uint8* data, uint err[2] )
{
    for( int i=0; i<8; i++ )
    {
        uint v = *data++;
        err[0] += v;
        err[1] += v*v;
    }
}

static uint CalcError( const uint block[2], uint average )
{
    uint err = block[1];
    err -= block[0] * 2 * average;
    err += 8 * sq( average );
    return err;
}

static void ProcessAverages( uint* a )
{
    for( int i=0; i<2; i++ )
    {
        int c1 = mul8bit( a[i*2+1], 31 );
        int c2 = mul8bit( a[i*2], 31 );

        int diff = c2 - c1;
        if( diff > 3 ) diff = 3;
        else if( diff < -4 ) diff = -4;

        int co = c1 + diff;

        a[5+i*2] = ( c1 << 3 ) | ( c1 >> 2 );
        a[4+i*2] = ( co << 3 ) | ( co >> 2 );
    }
    for( int i=0; i<4; i++ )
    {
        a[i] = g_avg2[mul8bit( a[i], 15 )];
    }
}

static void EncodeAverages( uint64& _d, const uint* a, size_t idx )
{
    auto d = _d;
    d |= ( idx << 24 );
    size_t base = idx << 1;

    uint v;
    if( ( idx & 0x2 ) == 0 )
    {
        v = ( a[base+0] >> 4 ) | ( a[base+1] & 0xF0 );
    }
    else
    {
        v = a[base+1] & 0xF8;
        int32 c = ( ( a[base+0] & 0xF8 ) - ( a[base+1] & 0xF8 ) ) >> 3;
        v |= c & ~0xFFFFFFF8;
    }
    d |= v | ( v << 8 ) | ( v << 16 );
    _d = d;
}

uint64 ProcessAlpha( const uint8* src )
{
    uint64 d = 0;

    {
        bool solid = true;
        const uint8* ptr = src + 1;
        for( int i=1; i<16; i++ )
        {
            if( *src != *ptr++ )
            {
                solid = false;
                break;
            }
        }
        if( solid )
        {
            uint c = *src & 0xF8;
            d |= 0x02000000 | ( c << 16 ) | ( c << 8 ) | c;
            return d;
        }
    }

    uint8 b23[2][8];
    const uint8* b[4] = { src+8, src, b23[0], b23[1] };

    for( int i=0; i<4; i++ )
    {
        *(b23[1]+i*2) = *(src+i*4);
        *(b23[0]+i*2) = *(src+i*4+3);
    }

    uint a[8];
    for( int i=0; i<4; i++ )
    {
        a[i] = Average1( b[i] );
    }
    ProcessAverages( a );

    uint err[4] = {};
    for( int i=0; i<4; i++ )
    {
        uint errblock[2] = {};
        CalcErrorBlock( b[i], errblock );
        err[i/2] += CalcError( errblock, a[i] );
        err[2+i/2] += CalcError( errblock, a[i+4] );
    }
    size_t idx = GetLeastError( err, 4 );

    EncodeAverages( d, a, idx );

    uint terr[2][8] = {};
    uint16 tsel[16][8];
    auto id = g_id[idx];
    const uint8* data = src;
    for( size_t i=0; i<16; i++ )
    {
        uint16* sel = tsel[i];
        uint bid = id[i];
        uint* ter = terr[bid%2];

        uint8 c = *data++;
#ifdef __SSE4_1__
        __m128i pix = _mm_set1_epi16(a[bid] - c);
        // Taking the absolute value is way faster. The values are only used to sort, so the result will be the same.
        __m128i error0 = _mm_abs_epi16(_mm_add_epi16(pix, g_table_SIMD[0]));
        __m128i error1 = _mm_abs_epi16(_mm_add_epi16(pix, g_table_SIMD[1]));
        __m128i error2 = _mm_abs_epi16(_mm_sub_epi16(pix, g_table_SIMD[0]));
        __m128i error3 = _mm_abs_epi16(_mm_sub_epi16(pix, g_table_SIMD[1]));

        __m128i index0 = _mm_and_si128(_mm_cmplt_epi16(error1, error0), _mm_set1_epi16(1));
        __m128i minError0 = _mm_min_epi16(error0, error1);

        __m128i index1 = _mm_sub_epi16(_mm_set1_epi16(2), _mm_cmplt_epi16(error3, error2));
        __m128i minError1 = _mm_min_epi16(error2, error3);

        __m128i minIndex = _mm_blendv_epi8(index0, index1, _mm_cmplt_epi16(minError1, minError0));
        __m128i minError = _mm_min_epi16(minError0, minError1);

        // Squaring the minimum error to produce correct values when adding
        __m128i squareErrorLo = _mm_mullo_epi16(minError, minError);
        __m128i squareErrorHi = _mm_mulhi_epi16(minError, minError);

        __m128i squareErrorLow = _mm_unpacklo_epi16(squareErrorLo, squareErrorHi);
        __m128i squareErrorHigh = _mm_unpackhi_epi16(squareErrorLo, squareErrorHi);

        squareErrorLow = _mm_add_epi32(squareErrorLow, _mm_lddqu_si128(((__m128i*)ter) + 0));
        _mm_storeu_si128(((__m128i*)ter) + 0, squareErrorLow);
        squareErrorHigh = _mm_add_epi32(squareErrorHigh, _mm_lddqu_si128(((__m128i*)ter) + 1));
        _mm_storeu_si128(((__m128i*)ter) + 1, squareErrorHigh);

        _mm_storeu_si128((__m128i*)sel, minIndex);
#else
        int32 pix = a[bid] - c;

        for( int t=0; t<8; t++ )
        {
            const int32* tab = g_table[t];
            uint idx = 0;
            uint err = sq( tab[0] + pix );
            for( int j=1; j<4; j++ )
            {
                uint local = sq( tab[j] + pix );
                if( local < err )
                {
                    err = local;
                    idx = j;
                }
            }
            *sel++ = idx;
            *ter++ += err;
        }
#endif
    }

    return FixByteOrder( EncodeSelectors( d, terr, tsel, id ) );
}

uint64 CheckSolidAlpha( const uint8* src )
{
#if __ARM_NEON__
    uint8x16_t d = vld1q_u8(src);
    uint8x16_t c = vdupq_n_u8(src[0]);
    uint8x16_t e = vceqq_u8(d, c);
    int64x2_t m = vreinterpretq_s64_u8(e);

    if (m[0] != -1 || m[1] != -1)
    {
        return ~(uint64)0;
    }
#else
    const uint8* ptr = src + 1;
    for( int i=1; i<16; i++ )
    {
        if( memcmp( src, ptr, 1 ) != 0 )
        {
            return ~(uint64)0;
        }
        ptr += 1;
    }
#endif
    return src[0];
}

uint64 ProcessAlpha_ETC2( const uint8* src )
{
    uint64 d = CheckSolidAlpha( src );
    if( d != ~(uint64)0 ) return d;

    uint8 min = src[0], max = src[0];
    uint16 sum = src[0];
    for( int i = 1; i < 16; ++i )
    {
        uint8 v = src[i];
        if( min > v )
            min = v;
        if( max < v )
            max = v;
        sum += v;
    }
    uint8 avg = sum / 16;
    uint8 diff = std::max(max - avg, avg - min);
#if __ARM_NEON__
    int16x8_t simdAvg = vdupq_n_s16(avg);
#endif

    int32 mod[16] =
    {
        (diff + 13) / 14, (diff + 11) / 12, (diff + 11) / 12, (diff + 11) / 12,
        (diff + 10) / 11, (diff + 9) / 10,  (diff + 9) / 10,  (diff + 9) / 10,
        (diff + 8) / 9,   (diff + 8) / 9,   (diff + 8) / 9,   (diff + 8) / 9,
        (diff + 8) / 9,   (diff + 8) / 9,   (diff + 7) / 8,   (diff + 7) / 8
    };
    uint8 indices[16][16];
    uint16 bestTable = 0;

    uint16 minTotalError = USHRT_MAX;
    for( int i = 0; i < 16; ++i )
    {
        uint16 totalError = 0;
#if __ARM_NEON__
        int16x8_t simdMod = vdupq_n_s16(mod[i]);
#endif
        for( int j = 0; j < 16; ++j )
        {
            uint16 error = USHRT_MAX;
#if __ARM_NEON__
            int16x8_t simdTable = vld1q_s16(g_tableAlpha[i]);
            int16x8_t simdValue = vmlaq_s16(simdAvg, simdMod, simdTable);
            uint8x8_t simdZip = vqmovun_s16(simdValue);
            uint8x8_t simdSrc = vdup_n_u8(src[j]);
            uint8x8_t simdDiff = vabd_u8(simdZip, simdSrc);
            for( int k = 0; k < 8; ++k )
            {
                uint16 d = simdDiff[k];
                if (error > d)
                {
                    error = d;
                    indices[i][j] = k;
                }
            }
#else
            for( int k = 0; k < 8; ++k )
            {
                int16 v = NiClamp(mod[i] * g_tableAlpha[i][k] + avg, 0, 255);
                uint16 d = NiAbs(v - src[j]);
                if (error > d)
                {
                    error = d;
                    indices[i][j] = k;
                }
            }
#endif
            totalError += error;
        }
        if (minTotalError > totalError)
        {
            minTotalError = totalError;
            bestTable = i;
        }
    }

    d = 0;
    for( int i = 0; i < 4; ++i )
    {
        d |= uint64(indices[bestTable][i * 4 + 0]) << (45 - i * 3);
        d |= uint64(indices[bestTable][i * 4 + 1]) << (33 - i * 3);
        d |= uint64(indices[bestTable][i * 4 + 2]) << (21 - i * 3);
        d |= uint64(indices[bestTable][i * 4 + 3]) << (9  - i * 3);
    }
    d |= uint64(bestTable)        << 48;
    d |= uint64(mod[bestTable])   << 52;
    d |= uint64(avg)              << 56;
    d = _bswap64(d);

    return d;
}
