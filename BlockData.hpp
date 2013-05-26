#ifndef __BLOCKDATA_HPP__
#define __BLOCKDATA_HPP__

#include <memory>

#include "BlockBitmap.hpp"
#include "Bitmap.hpp"
#include "Types.hpp"
#include "Vector.hpp"

class BlockData
{
public:
    BlockData( const BlockBitmapPtr& bitmap );
    ~BlockData();

    BitmapPtr Decode();

private:
    uint8* m_data;
    v2i m_size;
};

typedef std::shared_ptr<BlockData> BlockDataPtr;

#endif