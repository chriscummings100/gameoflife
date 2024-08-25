RWTexture2D g_output;
float2 g_draw_offset;
float g_draw_scale;

int2 g_game_dimensions;
StructuredBuffer<int> g_game_in;
RWStructuredBuffer<int> g_game_out;

int g_draw_mode;

int buffer_idx(int2 coord)
{
    return coord.y * g_game_dimensions.x + coord.x;
}
bool is_alive(int2 coord)
{
    return g_game_in[buffer_idx(coord)] == 1;
}

int count_neighbours(int2 coord) 
{
         int alive_neighbours = 0;
        if(is_alive(coord+int2(-1,-1)))  alive_neighbours++;
        if(is_alive(coord+int2(-1, 0)))  alive_neighbours++;
        if(is_alive(coord+int2(-1, 1)))  alive_neighbours++;
        if(is_alive(coord+int2( 0,-1)))  alive_neighbours++;
        if(is_alive(coord+int2( 0, 1)))  alive_neighbours++;
        if(is_alive(coord+int2( 1,-1)))  alive_neighbours++;
        if(is_alive(coord+int2( 1, 0)))  alive_neighbours++;
        if(is_alive(coord+int2( 1, 1)))  alive_neighbours++;
        return alive_neighbours;   
}

[shader("compute")]
[numthreads(16,16,1)]
void update(uint2 tid: SV_DispatchThreadID)
{
    if(any(tid >= g_game_dimensions))
        return;

    if(any(tid==0) || any(tid==g_game_dimensions-1))
    {
        g_game_out[buffer_idx(tid)] = 0;   
    }
    else 
    {
        int alive_neighbours = count_neighbours(tid);

        bool alive = is_alive(tid);
        if(alive)
        {
            if(alive_neighbours < 2)   
                alive = false;
            else if(alive_neighbours > 3)
                alive = false;
        }
        else 
        {
            if(alive_neighbours == 3)
                alive = true;
        }
        g_game_out[buffer_idx(tid)] = alive ? 1 : 0;   

    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint2 dims;
    uint2 pixel = tid.xy;
    g_output.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    float2 coord = pixel;
    coord -= float2(dims)*0.5;
    coord /= g_draw_scale;
    coord += g_draw_offset;
    coord += float2(g_game_dimensions) * 0.5;


    int2 game_coord = int2(round(coord));

    float4 bg = float4(0,0,0,1);
    if(any(game_coord%16==0))
        bg = float4(0.05,0.05,0.05,1);

    float4 fg = float4(1,1,1,1);
    if(count_neighbours(game_coord) < 2)
        fg = float4(1,0.75,0.75,1);

    if(all(game_coord >= 0) && all(game_coord < g_game_dimensions))
        g_output[pixel] = is_alive(game_coord) ? fg : bg;
    else 
        g_output[pixel] = bg;
}