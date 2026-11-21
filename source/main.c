// Roam3DS - step 5: real car collisions, damage, police pursuit
// City generated from hashes of chunk coordinates (64 m grid, roads on chunk borders).
// Civilians drive in lanes until hit; police chase the player, ram, and can be wrecked.
// Bottom screen: minimap (X = heading-up / north-up, L/R = zoom), heat and health HUD.
// SELECT = repair the car.

#include <3ds.h>
#include <citro3d.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "vshader_shbin.h"

#define CLEAR_COLOR 0x87CEEBFF // sky blue (RGBA)
#define MAP_CLEAR   0x121721FF // minimap background (RGBA)

#define DISPLAY_TRANSFER_FLAGS \
	(GX_TRANSFER_FLIP_VERT(0) | GX_TRANSFER_OUT_TILED(0) | GX_TRANSFER_RAW_COPY(0) | \
	GX_TRANSFER_IN_FORMAT(GX_TRANSFER_FMT_RGBA8) | GX_TRANSFER_OUT_FORMAT(GX_TRANSFER_FMT_RGB8) | \
	GX_TRANSFER_SCALING(GX_TRANSFER_SCALE_NO))

// ---- world / streaming settings ---------------------------------------------
#define CHUNK_SIZE       64.0f
#define ROAD_HALF        6.0f    // street half width
#define AVE_HALF         8.0f    // avenue half width
#define FWY_HALF         9.0f    // freeway half width
#define FWY_H            8.0f    // elevated freeway deck height
#define WATER_Y          (-2.2f)
#define TUN_BASE         9.5f    // tunnel mound half width at ground
#define TUN_TOP          5.5f    // ... and at the top
#define TUN_H            7.0f
#define TUN_OPEN_H       5.0f    // tunnel ceiling height
#define TUN_END          8.0f    // tunnel starts this far inside the chunk
#define MAX_CHUNKS       64
#define MAX_CHUNK_VERTS  6144
#define MAX_SOLIDS       10
#define SCAN_RADIUS      6

#define KEEP_AHEAD       200.0f
#define KEEP_BEHIND      60.0f
#define KEEP_CAR_R       100.0f
#define UNLOAD_MARGIN    40.0f

// ---- cars -------------------------------------------------------------------
#define MAX_CIV          6       // civilian cars (fewer than before)
#define MAX_COP          5       // police cars at 5 stars
#define MAX_CARS         12
#define CAR_R            2.0f
#define NUM_MESHES       10      // 0 player, 1..6 civilians, 7 wreck, 8/9 police (light phases)
#define SHOW_TRAFFIC_ON_MAP 0    // 1 = also show civilian cars on the minimap

#define PI 3.14159265f

typedef struct { float x, y, z; float r, g, b, a; } Vertex;
typedef struct { float x, y, z; } P3;

enum { B_NONE = 0, B_BLOCK, B_TOWER, B_STEPPED, B_HOUSE, B_WARE, B_PARK, B_SOLID };

typedef struct {
	float x0, z0, x1, z1, h;   // footprint and total height (h <= 0: not solid)
	float r, g, b;
	float hp;                   // podium / wall height for multi-part types
	int type, roof;
} Building;

typedef struct {
	int cx, cz;
	bool used;
	int nverts;
	Vertex* verts;
} Chunk;

typedef struct {
	float carX, carZ, camX, camZ, dirX, dirZ, keepAhead;
} StreamCtx;

enum { K_CIV = 0, K_COP };

typedef struct {
	bool active;
	int kind;
	float x, z, heading, speed;   // speed is along the heading
	float vx, vz, spin;           // sideways slide and yaw spin from impacts
	float hp;                     // 0..100
	float wreckT;                 // > 0 once wrecked (seconds since)
	float lostT;                  // time since knocked out of lane / retiring
	float cruise, lane, stuck;    // civilian lane driving
	int dir, color;
	bool lost;                    // civilian knocked out of its lane
} Car;

static Chunk chunks[MAX_CHUNKS];
static int loadedCount = 0;

static Car cars[MAX_CARS];
static Car player;
static float heat = 0.0f, bustedT = 0.0f, hitCool = 0.0f;
static int copWrecks = 0, gCopPhase = 0, gCivCount = 0, gCopCount = 0;
static float gNearestCop = 1e9f;

static Vertex* carMesh[NUM_MESHES];
static int carMeshN[NUM_MESHES];

static const float trafficPal[6][3] = {
	{0.90f,0.90f,0.90f}, {0.15f,0.25f,0.70f}, {0.65f,0.67f,0.70f},
	{0.10f,0.10f,0.12f}, {0.10f,0.45f,0.20f}, {0.90f,0.50f,0.10f}
};

// directions: 0 = north (-z), 1 = east (+x), 2 = south (+z), 3 = west (-x)
static const float dxs[4] = { 0, 1, 0, -1 };
static const float dzs[4] = { -1, 0, 1, 0 };
static const float laneOff[4] = { 3, 3, -3, -3 };            // right-hand traffic
static const float dirHeading[4] = { 0, PI / 2, PI, 3 * PI / 2 };

// ---- small helpers -----------------------------------------------------------
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int imod(int a, int m) { int r = a % m; return r < 0 ? r + m : r; }

static uint32_t hash2(int x, int z)
{
	uint32_t h = (uint32_t)x * 374761393u + (uint32_t)z * 668265263u;
	h = (h ^ (h >> 13)) * 1274126177u;
	return h ^ (h >> 16);
}

static float hrand(uint32_t* s)
{
	uint32_t x = *s;
	x ^= x << 13; x ^= x >> 17; x ^= x << 5;
	*s = x;
	return (x & 0xFFFFFF) / 16777216.0f;
}

static uint32_t rngState = 0x1234567u;
static float frand(void) { return hrand(&rngState); }

static float angleDiff(float a, float b)
{
	float d = fmodf(b - a + PI, 2.0f * PI);
	if (d < 0) d += 2.0f * PI;
	return d - PI;
}

static P3 p3(float x, float y, float z) { P3 p = { x, y, z }; return p; }

// ---- world rules (what is where) ----------------------------------------------
// N-S road lines are x = i * 64, E-W road lines are z = j * 64.
static bool isFreewayX(int i) { return imod(i, 4) == 2; }
static bool isAvenueX(int i)  { return !isFreewayX(i) && imod(i, 3) == 0; }
static bool isAvenueZ(int j)  { return imod(j, 3) == 0; }
static float roadHalfX(int i) { return isFreewayX(i) ? FWY_HALF : (isAvenueX(i) ? AVE_HALF : ROAD_HALF); }
static float roadHalfZ(int j) { return isAvenueZ(j) ? AVE_HALF : ROAD_HALF; }
static bool isRiverRow(int cz) { return imod(cz, 7) == 3; }

// freeway deck height at the crossing of freeway line i and E-W line j
static float fwyCrossH(int i, int j)
{
	return (hash2(i * 31 + 7, j * 17 + 3) % 5 < 3) ? FWY_H : 0.0f;
}

// plain streets (not avenue, not freeway) sometimes run through a tunnel
static bool tunnelOnLine(int i, int cz)
{
	return !isFreewayX(i) && !isAvenueX(i) && !isRiverRow(cz) &&
	       (hash2(i * 13 + 5, cz * 29 + 11) % 8) == 0;
}

// height of the freeway deck along line i at world z (flat near intersections, ramp between)
static float fwyHeightAt(int i, float z)
{
	int j = (int)floorf(z / CHUNK_SIZE);
	float d = z - j * CHUNK_SIZE;
	float t = clampf((d - 10.0f) / 44.0f, 0.0f, 1.0f);
	float a = fwyCrossH(i, j), b = fwyCrossH(i, j + 1);
	return a + (b - a) * t;
}

// which surface a car at height curY is driving on (ground or freeway deck)
static float surfaceY(float x, float z, float curY)
{
	int i = (int)floorf(x / CHUNK_SIZE + 0.5f);
	if (isFreewayX(i) && fabsf(x - i * CHUNK_SIZE) < FWY_HALF) {
		float h = fwyHeightAt(i, z);
		if (fabsf(h - curY) < 1.6f) return h;
	}
	return 0.0f;
}

// ---- chunk mesh building (scratch buffer, then copied to linear memory) --------
static Vertex scratch[MAX_CHUNK_VERTS];
static int vn;

static void pushV(P3 p, float r, float g, float b)
{
	Vertex* v = &scratch[vn++];
	v->x = p.x; v->y = p.y; v->z = p.z;
	v->r = r; v->g = g; v->b = b; v->a = 1.0f;
}

static void quad(P3 a, P3 b, P3 c, P3 d, float r, float g, float bl)
{
	if (vn + 6 > MAX_CHUNK_VERTS) return;
	pushV(a, r, g, bl); pushV(b, r, g, bl); pushV(c, r, g, bl);
	pushV(a, r, g, bl); pushV(c, r, g, bl); pushV(d, r, g, bl);
}

static void tri(P3 a, P3 b, P3 c, float r, float g, float bl)
{
	if (vn + 3 > MAX_CHUNK_VERTS) return;
	pushV(a, r, g, bl); pushV(b, r, g, bl); pushV(c, r, g, bl);
}

static void flatQuad(float x0, float z0, float x1, float z1, float y, float r, float g, float b)
{
	quad(p3(x0, y, z0), p3(x1, y, z0), p3(x1, y, z1), p3(x0, y, z1), r, g, b);
}

// box with 4 walls and a top (no bottom)
static void solidBox(float x0, float y0, float z0, float x1, float y1, float z1,
                     float r, float g, float b)
{
	quad(p3(x0,y0,z0), p3(x1,y0,z0), p3(x1,y1,z0), p3(x0,y1,z0), r*0.85f, g*0.85f, b*0.85f);
	quad(p3(x1,y0,z0), p3(x1,y0,z1), p3(x1,y1,z1), p3(x1,y1,z0), r*0.95f, g*0.95f, b*0.95f);
	quad(p3(x1,y0,z1), p3(x0,y0,z1), p3(x0,y1,z1), p3(x1,y1,z1), r, g, b);
	quad(p3(x0,y0,z1), p3(x0,y0,z0), p3(x0,y1,z0), p3(x0,y1,z1), r*0.75f, g*0.75f, b*0.75f);
	quad(p3(x0,y1,z0), p3(x1,y1,z0), p3(x1,y1,z1), p3(x0,y1,z1), r*0.7f, g*0.7f, b*0.7f);
}

static void wallPts(float x0, float z0, float x1, float z1, int side,
                    float* ax, float* az, float* bx, float* bz, float* nx, float* nz)
{
	switch (side) {
		case 0:  *ax = x0; *az = z0; *bx = x1; *bz = z0; *nx = 0;  *nz = -1; break;
		case 1:  *ax = x1; *az = z0; *bx = x1; *bz = z1; *nx = 1;  *nz = 0;  break;
		case 2:  *ax = x1; *az = z1; *bx = x0; *bz = z1; *nx = 0;  *nz = 1;  break;
		default: *ax = x0; *az = z1; *bx = x0; *bz = z0; *nx = -1; *nz = 0;  break;
	}
}

// four walls from yb to yt; style 0 = plain, 1 = window bands, 2 = glass curtain wall
static void addWalls(float x0, float z0, float x1, float z1, float yb, float yt,
                     float r, float g, float b, uint32_t* ws, int style)
{
	static const float shade[4] = { 0.85f, 0.95f, 1.0f, 0.75f };
	for (int side = 0; side < 4; side++) {
		float ax, az, bx, bz, nx, nz;
		wallPts(x0, z0, x1, z1, side, &ax, &az, &bx, &bz, &nx, &nz);
		float sh = shade[side];
		quad(p3(ax, yb, az), p3(bx, yb, bz), p3(bx, yt, bz), p3(ax, yt, az), r * sh, g * sh, b * sh);
		if (style == 0) continue;

		float len = fabsf(bx - ax) + fabsf(bz - az);
		float ux = (bx - ax) / len, uz = (bz - az) / len;
		float sx = ax + ux * 1.2f + nx * 0.06f, sz = az + uz * 1.2f + nz * 0.06f;
		float ex = bx - ux * 1.2f + nx * 0.06f, ez = bz - uz * 1.2f + nz * 0.06f;
		float sp = fmaxf(4.0f, (yt - yb - 2.0f) / 8.0f);   // at most ~8 bands per wall
		float wh = (style == 2) ? sp * 0.72f : 2.2f;
		for (float y0 = yb + 1.2f; y0 + wh < yt - 0.6f; y0 += sp) {
			float gv = hrand(ws), wr, wg, wb;
			if (style == 2) {
				if (gv < 0.12f)      { wr = 0.85f; wg = 0.88f; wb = 0.68f; }
				else if (gv < 0.55f) { wr = 0.38f; wg = 0.58f; wb = 0.78f; }
				else                 { wr = 0.24f; wg = 0.40f; wb = 0.60f; }
			} else {
				if (gv < 0.15f)      { wr = 0.75f; wg = 0.78f; wb = 0.55f; }
				else if (gv < 0.5f)  { wr = 0.20f; wg = 0.28f; wb = 0.40f; }
				else                 { wr = 0.12f; wg = 0.17f; wb = 0.26f; }
			}
			quad(p3(sx, y0, sz), p3(ex, y0, ez), p3(ex, y0 + wh, ez), p3(sx, y0 + wh, sz),
			     wr * sh, wg * sh, wb * sh);
		}
	}
}

static void addTree(float x, float z, float s, uint32_t* ws)
{
	float v = 0.85f + hrand(ws) * 0.3f;
	solidBox(x - 0.25f * s, 0.0f, z - 0.25f * s, x + 0.25f * s, 1.6f * s, z + 0.25f * s,
	         0.38f, 0.26f, 0.16f);
	float r = 1.7f * s, y0 = 1.3f * s, y1 = 4.6f * s;
	P3 a = p3(x - r, y0, z - r), b = p3(x + r, y0, z - r);
	P3 c = p3(x + r, y0, z + r), d = p3(x - r, y0, z + r), t = p3(x, y1, z);
	tri(a, b, t, 0.16f * v, 0.42f * v, 0.17f * v);
	tri(b, c, t, 0.20f * v, 0.50f * v, 0.20f * v);
	tri(c, d, t, 0.14f * v, 0.36f * v, 0.15f * v);
	tri(d, a, t, 0.18f * v, 0.46f * v, 0.19f * v);
}

// ---- buildings -------------------------------------------------------------------
// Deterministic per chunk: 2x2 lots, each lot gets a type based on the district.
static void getBuildings(int cx, int cz, Building out[4])
{
	static const float palBlock[6][3] = {
		{0.62f,0.58f,0.52f}, {0.48f,0.52f,0.58f}, {0.70f,0.66f,0.60f},
		{0.45f,0.40f,0.38f}, {0.58f,0.62f,0.60f}, {0.66f,0.50f,0.42f}
	};
	static const float palGlass[4][3] = {
		{0.32f,0.45f,0.58f}, {0.30f,0.52f,0.55f}, {0.40f,0.48f,0.62f}, {0.25f,0.35f,0.45f}
	};
	static const float palStone[3][3] = {
		{0.70f,0.64f,0.55f}, {0.62f,0.58f,0.62f}, {0.66f,0.60f,0.50f}
	};
	static const float palHouse[5][3] = {
		{0.80f,0.72f,0.58f}, {0.72f,0.45f,0.38f}, {0.85f,0.82f,0.75f},
		{0.55f,0.65f,0.72f}, {0.62f,0.70f,0.52f}
	};
	// cumulative roll thresholds: tower, stepped, block, warehouse, house (rest = park)
	static const float distr[3][5] = {
		{ 0.38f, 0.60f, 0.88f, 0.88f, 0.88f },   // downtown
		{ 0.10f, 0.25f, 0.60f, 0.78f, 0.88f },   // middle ring
		{ 0.00f, 0.00f, 0.12f, 0.35f, 0.78f }    // outskirts
	};

	memset(out, 0, 4 * sizeof(Building));
	if (isRiverRow(cz)) return;

	uint32_t s = hash2(cx, cz) | 1u;
	float ox = cx * CHUNK_SIZE, oz = cz * CHUNK_SIZE;
	float dc = sqrtf((float)(cx * cx + cz * cz));
	float boost = 45.0f - 9.0f * dc;           // taller buildings near the origin
	if (boost < 0) boost = 0;
	int dist = dc < 3.5f ? 0 : (dc < 8.0f ? 1 : 2);

	int n = 0;
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 2; i++) {
			Building* b = &out[n++];
			float roll = hrand(&s);
			float rw = hrand(&s), rd = hrand(&s), ro1 = hrand(&s), ro2 = hrand(&s);
			float rh1 = hrand(&s), rh2 = hrand(&s), rc = hrand(&s), rt = hrand(&s), rr = hrand(&s);

			int type;
			if (roll < distr[dist][0])      type = B_TOWER;
			else if (roll < distr[dist][1]) type = B_STEPPED;
			else if (roll < distr[dist][2]) type = B_BLOCK;
			else if (roll < distr[dist][3]) type = B_WARE;
			else if (roll < distr[dist][4]) type = B_HOUSE;
			else                            type = B_PARK;

			float w, dep, h = 0.0f, hp = 0.0f, cr, cg, cb;
			switch (type) {
				case B_TOWER: {
					int k = (int)(rc * 4.0f) & 3;
					w = 14.0f + rw * 6.0f; dep = 14.0f + rd * 6.0f;
					h = 32.0f + rh1 * 28.0f + boost; hp = 8.0f + rh2 * 4.0f;
					cr = palGlass[k][0]; cg = palGlass[k][1]; cb = palGlass[k][2];
					break; }
				case B_STEPPED: {
					int k = (int)(rc * 3.0f) % 3;
					w = 16.0f + rw * 5.0f; dep = 16.0f + rd * 5.0f;
					h = 22.0f + rh1 * 18.0f + boost * 0.6f; hp = h * 0.5f;
					cr = palStone[k][0]; cg = palStone[k][1]; cb = palStone[k][2];
					break; }
				case B_WARE:
					w = 18.0f + rw * 4.0f; dep = 16.0f + rd * 6.0f;
					h = 7.0f + rh1 * 3.0f;
					cr = 0.55f; cg = 0.55f; cb = 0.56f;
					break;
				case B_HOUSE: {
					int k = (int)(rc * 5.0f) % 5;
					w = 9.0f + rw * 4.0f; dep = 9.0f + rd * 4.0f;
					hp = 3.4f; h = hp + 2.4f;
					cr = palHouse[k][0]; cg = palHouse[k][1]; cb = palHouse[k][2];
					break; }
				case B_PARK:
					w = 22.0f; dep = 22.0f; h = 0.0f;
					cr = 0.26f; cg = 0.54f; cb = 0.26f;
					break;
				default: {
					int k = (int)(rc * 6.0f) % 6;
					w = 12.0f + rw * 8.0f; dep = 12.0f + rd * 8.0f;
					h = 8.0f + rh1 * 16.0f + boost * 0.5f;
					cr = palBlock[k][0]; cg = palBlock[k][1]; cb = palBlock[k][2];
					break; }
			}
			float tone = 0.9f + rt * 0.2f;
			float cellx = ox + 10.0f + i * 22.0f, cellz = oz + 10.0f + j * 22.0f;
			b->x0 = cellx + ro1 * (22.0f - w); b->x1 = b->x0 + w;
			b->z0 = cellz + ro2 * (22.0f - dep); b->z1 = b->z0 + dep;
			b->h = h; b->hp = hp; b->type = type;
			b->roof = (int)(rr * 16.0f) & 15;
			b->r = cr * tone; b->g = cg * tone; b->b = cb * tone;
		}
	}
}

static void addBuilding(const Building* b, uint32_t* ws)
{
	float cx = (b->x0 + b->x1) * 0.5f, cz = (b->z0 + b->z1) * 0.5f;
	float x0 = b->x0, z0 = b->z0, x1 = b->x1, z1 = b->z1, h = b->h;

	switch (b->type) {
	case B_BLOCK:
		addWalls(x0, z0, x1, z1, 0.0f, h, b->r, b->g, b->b, ws, 1);
		flatQuad(x0, z0, x1, z1, h, b->r * 0.6f, b->g * 0.6f, b->b * 0.6f);
		if (b->roof & 1) solidBox(cx - 2, h, cz - 2, cx + 2, h + 1.8f, cz + 2, 0.5f, 0.5f, 0.52f);
		if (b->roof & 2) solidBox(x0 + 1.5f, h, z0 + 1.5f, x0 + 3.5f, h + 1.0f, z0 + 3.5f, 0.45f, 0.46f, 0.5f);
		break;

	case B_TOWER: {
		float hp = b->hp, in = 3.0f;
		addWalls(x0, z0, x1, z1, 0.0f, hp, b->r * 1.25f, b->g * 1.2f, b->b * 1.1f, ws, 1);
		flatQuad(x0, z0, x1, z1, hp, 0.30f, 0.30f, 0.32f);
		addWalls(x0 + in, z0 + in, x1 - in, z1 - in, hp, h, b->r, b->g, b->b, ws, 2);
		flatQuad(x0 + in, z0 + in, x1 - in, z1 - in, h, 0.25f, 0.27f, 0.30f);
		if (b->roof & 1) solidBox(cx - 0.25f, h, cz - 0.25f, cx + 0.25f, h + 9.0f, cz + 0.25f, 0.75f, 0.75f, 0.78f);
		if (b->roof & 2) solidBox(cx - 2.5f, h, cz - 2.5f, cx + 2.5f, h + 2.4f, cz + 2.5f, 0.45f, 0.48f, 0.52f);
		break; }

	case B_STEPPED: {
		float t1 = h * 0.5f, t2 = h * 0.8f;
		addWalls(x0, z0, x1, z1, 0.0f, t1, b->r, b->g, b->b, ws, 1);
		flatQuad(x0, z0, x1, z1, t1, b->r * 0.6f, b->g * 0.6f, b->b * 0.6f);
		addWalls(x0 + 2.5f, z0 + 2.5f, x1 - 2.5f, z1 - 2.5f, t1, t2, b->r * 1.05f, b->g * 1.05f, b->b * 1.05f, ws, 1);
		flatQuad(x0 + 2.5f, z0 + 2.5f, x1 - 2.5f, z1 - 2.5f, t2, b->r * 0.6f, b->g * 0.6f, b->b * 0.6f);
		addWalls(x0 + 5.0f, z0 + 5.0f, x1 - 5.0f, z1 - 5.0f, t2, h, b->r * 1.1f, b->g * 1.1f, b->b * 1.1f, ws, 1);
		flatQuad(x0 + 5.0f, z0 + 5.0f, x1 - 5.0f, z1 - 5.0f, h, b->r * 0.55f, b->g * 0.55f, b->b * 0.55f);
		if (b->roof & 1) solidBox(cx - 0.3f, h, cz - 0.3f, cx + 0.3f, h + 6.0f, cz + 0.3f, 0.7f, 0.7f, 0.72f);
		break; }

	case B_WARE:
		addWalls(x0, z0, x1, z1, 0.0f, h, b->r, b->g, b->b, ws, 0);
		flatQuad(x0, z0, x1, z1, h, b->r * 0.7f, b->g * 0.7f, b->b * 0.72f);
		// colored stripe under the roof line
		for (int side = 0; side < 4; side++) {
			float ax, az, bx, bz, nx, nz;
			wallPts(x0, z0, x1, z1, side, &ax, &az, &bx, &bz, &nx, &nz);
			quad(p3(ax + nx * 0.05f, h - 1.6f, az + nz * 0.05f), p3(bx + nx * 0.05f, h - 1.6f, bz + nz * 0.05f),
			     p3(bx + nx * 0.05f, h - 0.6f, bz + nz * 0.05f), p3(ax + nx * 0.05f, h - 0.6f, az + nz * 0.05f),
			     0.70f, 0.28f, 0.20f);
		}
		// loading doors on the north wall
		for (int k = 0; k < 3; k++) {
			float wl = x1 - x0, u0 = x0 + wl * (0.12f + 0.29f * k), u1 = u0 + wl * 0.2f;
			quad(p3(u0, 0.0f, z0 - 0.05f), p3(u1, 0.0f, z0 - 0.05f), p3(u1, 3.2f, z0 - 0.05f), p3(u0, 3.2f, z0 - 0.05f),
			     0.20f, 0.22f, 0.26f);
		}
		// skylights
		flatQuad(cx - 6.0f, cz - 3.0f, cx - 2.0f, cz + 3.0f, h + 0.03f, 0.32f, 0.45f, 0.55f);
		flatQuad(cx + 2.0f, cz - 3.0f, cx + 6.0f, cz + 3.0f, h + 0.03f, 0.32f, 0.45f, 0.55f);
		break;

	case B_HOUSE: {
		static const float roofPal[4][3] = {
			{0.50f,0.22f,0.18f}, {0.30f,0.30f,0.33f}, {0.42f,0.28f,0.20f}, {0.22f,0.30f,0.38f}
		};
		float hp = b->hp, rh = b->h - b->hp;
		const float* rc = roofPal[b->roof & 3];
		flatQuad(x0 - 4.0f, z0 - 4.0f, x1 + 4.0f, z1 + 4.0f, 0.02f, 0.30f, 0.52f, 0.27f);  // lawn
		addWalls(x0, z0, x1, z1, 0.0f, hp, b->r, b->g, b->b, ws, 0);
		for (int side = 0; side < 4; side++) {   // one window per wall
			float ax, az, bx, bz, nx, nz;
			wallPts(x0, z0, x1, z1, side, &ax, &az, &bx, &bz, &nx, &nz);
			float sx = ax + (bx - ax) * 0.28f + nx * 0.05f, sz = az + (bz - az) * 0.28f + nz * 0.05f;
			float ex = ax + (bx - ax) * 0.72f + nx * 0.05f, ez = az + (bz - az) * 0.72f + nz * 0.05f;
			quad(p3(sx, 1.0f, sz), p3(ex, 1.0f, ez), p3(ex, 2.3f, ez), p3(sx, 2.3f, sz), 0.16f, 0.22f, 0.30f);
		}
		if (x1 - x0 >= z1 - z0) {   // ridge along x
			float xa = x0 - 0.3f, xb = x1 + 0.3f, yr = hp + rh;
			quad(p3(xa, hp, z0 - 0.3f), p3(xb, hp, z0 - 0.3f), p3(xb, yr, cz), p3(xa, yr, cz), rc[0] * 0.9f, rc[1] * 0.9f, rc[2] * 0.9f);
			quad(p3(xb, hp, z1 + 0.3f), p3(xa, hp, z1 + 0.3f), p3(xa, yr, cz), p3(xb, yr, cz), rc[0], rc[1], rc[2]);
			tri(p3(x0, hp, z0), p3(x0, hp, z1), p3(x0, yr, cz), b->r * 0.8f, b->g * 0.8f, b->b * 0.8f);
			tri(p3(x1, hp, z0), p3(x1, hp, z1), p3(x1, yr, cz), b->r * 0.9f, b->g * 0.9f, b->b * 0.9f);
		} else {                     // ridge along z
			float za = z0 - 0.3f, zb = z1 + 0.3f, yr = hp + rh;
			quad(p3(x0 - 0.3f, hp, za), p3(x0 - 0.3f, hp, zb), p3(cx, yr, zb), p3(cx, yr, za), rc[0] * 0.9f, rc[1] * 0.9f, rc[2] * 0.9f);
			quad(p3(x1 + 0.3f, hp, zb), p3(x1 + 0.3f, hp, za), p3(cx, yr, za), p3(cx, yr, zb), rc[0], rc[1], rc[2]);
			tri(p3(x0, hp, z0), p3(x1, hp, z0), p3(cx, yr, z0), b->r * 0.85f, b->g * 0.85f, b->b * 0.85f);
			tri(p3(x0, hp, z1), p3(x1, hp, z1), p3(cx, yr, z1), b->r * 0.95f, b->g * 0.95f, b->b * 0.95f);
		}
		break; }

	case B_PARK:
		flatQuad(x0, z0, x1, z1, 0.025f, 0.26f, 0.54f, 0.26f);
		flatQuad(cx - 0.9f, z0, cx + 0.9f, z1, 0.035f, 0.72f, 0.66f, 0.52f);   // cross paths
		flatQuad(x0, cz - 0.9f, x1, cz + 0.9f, 0.036f, 0.72f, 0.66f, 0.52f);
		for (int k = 0; k < 4; k++) {
			float tx = cx + ((k & 1) ? 6.0f : -6.0f) + (hrand(ws) - 0.5f) * 2.0f;
			float tz = cz + ((k & 2) ? 6.0f : -6.0f) + (hrand(ws) - 0.5f) * 2.0f;
			addTree(tx, tz, 0.9f + hrand(ws) * 0.5f, ws);
		}
		if (b->roof & 1) {   // small fountain
			solidBox(cx - 1.5f, 0.0f, cz - 1.5f, cx + 1.5f, 0.5f, cz + 1.5f, 0.6f, 0.6f, 0.62f);
			flatQuad(cx - 1.2f, cz - 1.2f, cx + 1.2f, cz + 1.2f, 0.52f, 0.30f, 0.55f, 0.80f);
		}
		break;

	default:
		break;
	}
}

// collision solids for the chunk: buildings, tunnel mounds and the river
static int getSolids(int cx, int cz, Building out[MAX_SOLIDS])
{
	int n = 0;
	Building bl[4];
	getBuildings(cx, cz, bl);
	for (int i = 0; i < 4; i++)
		if (bl[i].h > 0.0f && bl[i].type != B_PARK && bl[i].type != B_NONE) out[n++] = bl[i];

	const float S = CHUNK_SIZE;
	float ox = cx * S, oz = cz * S;
	for (int e = 0; e < 2; e++) {
		int li = cx + e;
		if (!tunnelOnLine(li, cz)) continue;
		float lx = li * S;
		Building w; memset(&w, 0, sizeof w);
		w.type = B_SOLID; w.h = TUN_H; w.z0 = oz + TUN_END; w.z1 = oz + S - TUN_END;
		w.x0 = lx - TUN_BASE; w.x1 = lx - ROAD_HALF; out[n++] = w;
		w.x0 = lx + ROAD_HALF; w.x1 = lx + TUN_BASE; out[n++] = w;
	}
	if (isRiverRow(cz)) {
		Building w; memset(&w, 0, sizeof w);
		w.type = B_SOLID; w.h = 1.0f;
		w.x0 = ox + roadHalfX(cx); w.x1 = ox + S - roadHalfX(cx + 1);
		w.z0 = oz + roadHalfZ(cz) + 2.0f; w.z1 = oz + S - roadHalfZ(cz + 1) - 2.0f;
		out[n++] = w;
	}
	return n;
}

// ---- freeway (drawn by the chunk east of the line) -----------------------------------
static void addFreeway(int i, int cz, float lx, float oz, float baseY)
{
	const float FH = FWY_HALF;
	static const float brk[5] = { 0.0f, 10.0f, 32.0f, 54.0f, 64.0f };
	float a = fwyCrossH(i, cz), b = fwyCrossH(i, cz + 1), hz[5];
	for (int k = 0; k < 5; k++) hz[k] = a + (b - a) * clampf((brk[k] - 10.0f) / 44.0f, 0.0f, 1.0f);

	for (int k = 0; k < 4; k++) {
		float za = oz + brk[k], zb = oz + brk[k + 1];
		float ha = hz[k] + 0.06f, hb = hz[k + 1] + 0.06f;
		bool elev = (hz[k] + hz[k + 1]) > 0.8f;
		bool bar = (k == 1 || k == 2) || elev;   // no barriers through at-grade intersections

		// deck
		quad(p3(lx - FH, ha, za), p3(lx + FH, ha, za), p3(lx + FH, hb, zb), p3(lx - FH, hb, zb),
		     0.17f, 0.17f, 0.18f);

		if (elev) {   // girders and underside
			quad(p3(lx - FH, ha - 1.4f, za), p3(lx - FH, hb - 1.4f, zb), p3(lx - FH, hb, zb), p3(lx - FH, ha, za), 0.42f, 0.42f, 0.44f);
			quad(p3(lx + FH, ha - 1.4f, za), p3(lx + FH, hb - 1.4f, zb), p3(lx + FH, hb, zb), p3(lx + FH, ha, za), 0.42f, 0.42f, 0.44f);
			quad(p3(lx - FH, ha - 1.4f, za), p3(lx + FH, ha - 1.4f, za), p3(lx + FH, hb - 1.4f, zb), p3(lx - FH, hb - 1.4f, zb), 0.28f, 0.28f, 0.30f);
		}
		if (!bar) continue;

		// median barrier
		quad(p3(lx - 0.45f, ha + 0.5f, za), p3(lx + 0.45f, ha + 0.5f, za), p3(lx + 0.45f, hb + 0.5f, zb), p3(lx - 0.45f, hb + 0.5f, zb), 0.72f, 0.72f, 0.70f);
		quad(p3(lx - 0.45f, ha, za), p3(lx - 0.45f, hb, zb), p3(lx - 0.45f, hb + 0.5f, zb), p3(lx - 0.45f, ha + 0.5f, za), 0.55f, 0.55f, 0.54f);
		quad(p3(lx + 0.45f, ha, za), p3(lx + 0.45f, hb, zb), p3(lx + 0.45f, hb + 0.5f, zb), p3(lx + 0.45f, ha + 0.5f, za), 0.55f, 0.55f, 0.54f);

		// side barriers
		for (int sd = 0; sd < 2; sd++) {
			float sg = sd ? 1.0f : -1.0f;
			float xo = lx + sg * FH, xi = lx + sg * (FH - 0.5f), xm = lx + sg * (FH - 0.25f);
			quad(p3(xm, ha, za), p3(xm, hb, zb), p3(xm, hb + 1.0f, zb), p3(xm, ha + 1.0f, za), 0.60f, 0.60f, 0.58f);
			quad(p3(xo, ha + 1.0f, za), p3(xi, ha + 1.0f, za), p3(xi, hb + 1.0f, zb), p3(xo, hb + 1.0f, zb), 0.74f, 0.74f, 0.72f);
		}

		// lane dashes (middle pieces only)
		if (k == 1 || k == 2) {
			float zA = za + (zb - za) * 0.2f, zB = za + (zb - za) * 0.6f;
			float yA = ha + (hb - ha) * 0.2f + 0.03f, yB = ha + (hb - ha) * 0.6f + 0.03f;
			for (int sd = 0; sd < 2; sd++) {
				float xc = lx + (sd ? 4.5f : -4.5f);
				quad(p3(xc - 0.12f, yA, zA), p3(xc + 0.12f, yA, zA), p3(xc + 0.12f, yB, zB), p3(xc - 0.12f, yB, zB), 0.9f, 0.9f, 0.9f);
			}
		}
	}

	// support pillars at mid-segment
	if (hz[2] > 1.2f) {
		float zm = oz + 32.0f, top = hz[2] - 1.4f;
		solidBox(lx - 6.4f, baseY, zm - 0.9f, lx - 4.8f, top, zm + 0.9f, 0.45f, 0.45f, 0.47f);
		solidBox(lx + 4.8f, baseY, zm - 0.9f, lx + 6.4f, top, zm + 0.9f, 0.45f, 0.45f, 0.47f);
		solidBox(lx - 6.4f, top - 0.8f, zm - 0.7f, lx + 6.4f, top, zm + 0.7f, 0.40f, 0.40f, 0.42f);
	}
}

// ---- tunnel: grassy mound over a street, portals at both ends ----------------------------
static void addTunnel(float lx, float oz)
{
	float z0 = oz + TUN_END, z1 = oz + CHUNK_SIZE - TUN_END;
	const float Ow = ROAD_HALF - 0.6f, OH = TUN_OPEN_H;
	float sh = TUN_BASE - (TUN_BASE - TUN_TOP) * (OH / TUN_H);   // mound half width at ceiling height

	// mound slopes and top
	quad(p3(lx - TUN_BASE, 0.0f, z0), p3(lx - TUN_BASE, 0.0f, z1), p3(lx - TUN_TOP, TUN_H, z1), p3(lx - TUN_TOP, TUN_H, z0), 0.27f, 0.45f, 0.23f);
	quad(p3(lx + TUN_BASE, 0.0f, z1), p3(lx + TUN_BASE, 0.0f, z0), p3(lx + TUN_TOP, TUN_H, z0), p3(lx + TUN_TOP, TUN_H, z1), 0.29f, 0.48f, 0.25f);
	quad(p3(lx - TUN_TOP, TUN_H, z0), p3(lx + TUN_TOP, TUN_H, z0), p3(lx + TUN_TOP, TUN_H, z1), p3(lx - TUN_TOP, TUN_H, z1), 0.34f, 0.54f, 0.29f);

	// concrete portals
	for (int e = 0; e < 2; e++) {
		float z = e ? z1 : z0;
		quad(p3(lx - TUN_BASE, 0.0f, z), p3(lx - Ow, 0.0f, z), p3(lx - Ow, OH, z), p3(lx - sh, OH, z), 0.58f, 0.58f, 0.56f);
		quad(p3(lx + Ow, 0.0f, z), p3(lx + TUN_BASE, 0.0f, z), p3(lx + sh, OH, z), p3(lx + Ow, OH, z), 0.58f, 0.58f, 0.56f);
		quad(p3(lx - sh, OH, z), p3(lx + sh, OH, z), p3(lx + TUN_TOP, TUN_H, z), p3(lx - TUN_TOP, TUN_H, z), 0.62f, 0.62f, 0.60f);
	}

	// dark interior: walls and ceiling
	quad(p3(lx - Ow, 0.0f, z0), p3(lx - Ow, 0.0f, z1), p3(lx - Ow, OH, z1), p3(lx - Ow, OH, z0), 0.14f, 0.14f, 0.16f);
	quad(p3(lx + Ow, 0.0f, z0), p3(lx + Ow, 0.0f, z1), p3(lx + Ow, OH, z1), p3(lx + Ow, OH, z0), 0.14f, 0.14f, 0.16f);
	quad(p3(lx - Ow, OH, z0), p3(lx + Ow, OH, z0), p3(lx + Ow, OH, z1), p3(lx - Ow, OH, z1), 0.10f, 0.10f, 0.12f);

	// ceiling lights
	for (float z = z0 + 6.0f; z < z1 - 3.0f; z += 8.0f)
		flatQuad(lx - 0.6f, z, lx + 0.6f, z + 1.5f, OH - 0.02f, 1.0f, 0.92f, 0.60f);
}

// ---- chunks ------------------------------------------------------------------------
static Chunk* findChunk(int cx, int cz)
{
	for (int i = 0; i < MAX_CHUNKS; i++)
		if (chunks[i].used && chunks[i].cx == cx && chunks[i].cz == cz)
			return &chunks[i];
	return NULL;
}

static bool loadChunk(int cx, int cz)
{
	Chunk* slot = NULL;
	for (int i = 0; i < MAX_CHUNKS; i++)
		if (!chunks[i].used) { slot = &chunks[i]; break; }
	if (!slot) return false;

	vn = 0;
	const float S = CHUNK_SIZE;
	float ox = cx * S, oz = cz * S;
	float Rw = roadHalfX(cx), Re = roadHalfX(cx + 1);
	float Rn = roadHalfZ(cz), Rs = roadHalfZ(cz + 1);
	bool river = isRiverRow(cz);
	float dc = sqrtf((float)(cx * cx + cz * cz));
	float gr, gg, gb;
	if (dc < 3.5f)      { gr = 0.55f; gg = 0.55f; gb = 0.53f; }   // downtown paving
	else if (dc < 8.0f) { gr = 0.50f; gg = 0.52f; gb = 0.48f; }
	else                { gr = 0.40f; gg = 0.50f; gb = 0.36f; }   // grassy outskirts
	const float ar = 0.17f, ag = 0.17f, ab = 0.18f;
	float wz0 = oz + Rn + 2.0f, wz1 = oz + S - Rs - 2.0f;

	// ground (or river)
	if (!river) {
		flatQuad(ox, oz, ox + S, oz + S, 0.0f, gr, gg, gb);
	} else {
		flatQuad(ox, oz, ox + S, wz0, 0.0f, gr, gg, gb);
		flatQuad(ox, wz1, ox + S, oz + S, 0.0f, gr, gg, gb);
		flatQuad(ox, wz0, ox + S, wz1, WATER_Y, 0.12f, 0.30f, 0.50f);
		uint32_t sd = hash2(cx, cz + 99) | 1u;
		for (int k = 0; k < 5; k++) {
			float sx = ox + hrand(&sd) * (S - 10.0f);
			float sz = wz0 + 3.0f + hrand(&sd) * (wz1 - wz0 - 6.0f);
			flatQuad(sx, sz, sx + 8.0f, sz + 0.5f, WATER_Y + 0.02f, 0.20f, 0.42f, 0.64f);
		}
		quad(p3(ox, 0.0f, wz0), p3(ox + S, 0.0f, wz0), p3(ox + S, WATER_Y, wz0), p3(ox, WATER_Y, wz0), 0.38f, 0.38f, 0.40f);
		quad(p3(ox, 0.0f, wz1), p3(ox + S, 0.0f, wz1), p3(ox + S, WATER_Y, wz1), p3(ox, WATER_Y, wz1), 0.34f, 0.34f, 0.36f);
	}

	// asphalt: this chunk's half of the road on each edge (freeways draw their own deck)
	flatQuad(ox, oz, ox + S, oz + Rn, 0.04f, ar, ag, ab);
	flatQuad(ox, oz + S - Rs, ox + S, oz + S, 0.04f, ar, ag, ab);
	if (!isFreewayX(cx))     flatQuad(ox, oz, ox + Rw, oz + S, 0.04f, ar, ag, ab);
	if (!isFreewayX(cx + 1)) flatQuad(ox + S - Re, oz, ox + S, oz + S, 0.04f, ar, ag, ab);

	// freeway on this chunk's west line
	if (isFreewayX(cx)) addFreeway(cx, cz, ox, oz, river ? WATER_Y : 0.0f);

	// markings on the west line (street dashes / avenue median)
	if (!isFreewayX(cx)) {
		if (isAvenueX(cx)) {
			flatQuad(ox - 0.7f, oz + Rn + 0.5f, ox + 0.7f, oz + S - Rs - 0.5f, 0.08f, 0.62f, 0.62f, 0.60f);
			flatQuad(ox - 0.45f, oz + Rn + 0.5f, ox + 0.45f, oz + S - Rs - 0.5f, 0.10f, 0.28f, 0.50f, 0.26f);
		} else {
			for (float t = 10.0f; t <= 50.0f; t += 8.0f)
				flatQuad(ox - 0.15f, oz + t, ox + 0.15f, oz + t + 4.0f, 0.08f, 0.9f, 0.8f, 0.2f);
		}
	}
	// markings on the north line
	if (isAvenueZ(cz)) {
		flatQuad(ox + Rw + 0.5f, oz - 0.7f, ox + S - Re - 0.5f, oz + 0.7f, 0.08f, 0.62f, 0.62f, 0.60f);
		flatQuad(ox + Rw + 0.5f, oz - 0.45f, ox + S - Re - 0.5f, oz + 0.45f, 0.10f, 0.28f, 0.50f, 0.26f);
	} else {
		for (float t = 10.0f; t <= 50.0f; t += 8.0f)
			flatQuad(ox + t, oz - 0.15f, ox + t + 4.0f, oz + 0.15f, 0.08f, 0.9f, 0.8f, 0.2f);
	}

	// tunnel on the west line
	if (tunnelOnLine(cx, cz)) addTunnel(ox, oz);

	// bridge details where N-S roads cross the river
	if (river) {
		float mid = (wz0 + wz1) * 0.5f;
		if (!isFreewayX(cx)) {
			float e0 = ox + Rw;
			quad(p3(e0, -0.9f, wz0), p3(e0, -0.9f, wz1), p3(e0, 0.04f, wz1), p3(e0, 0.04f, wz0), 0.42f, 0.42f, 0.44f);
			quad(p3(e0 - 0.2f, 0.04f, wz0), p3(e0 - 0.2f, 0.04f, wz1), p3(e0 - 0.2f, 1.1f, wz1), p3(e0 - 0.2f, 1.1f, wz0), 0.58f, 0.58f, 0.56f);
			quad(p3(e0 - 0.45f, 1.1f, wz0), p3(e0, 1.1f, wz0), p3(e0, 1.1f, wz1), p3(e0 - 0.45f, 1.1f, wz1), 0.70f, 0.70f, 0.68f);
			solidBox(ox - 1.0f, WATER_Y, mid - 1.0f, ox + 1.0f, -0.9f, mid + 1.0f, 0.48f, 0.48f, 0.50f);
		}
		if (!isFreewayX(cx + 1)) {
			float e1 = ox + S - Re;
			quad(p3(e1, -0.9f, wz0), p3(e1, -0.9f, wz1), p3(e1, 0.04f, wz1), p3(e1, 0.04f, wz0), 0.42f, 0.42f, 0.44f);
			quad(p3(e1 + 0.2f, 0.04f, wz0), p3(e1 + 0.2f, 0.04f, wz1), p3(e1 + 0.2f, 1.1f, wz1), p3(e1 + 0.2f, 1.1f, wz0), 0.58f, 0.58f, 0.56f);
			quad(p3(e1, 1.1f, wz0), p3(e1 + 0.45f, 1.1f, wz0), p3(e1 + 0.45f, 1.1f, wz1), p3(e1, 1.1f, wz1), 0.70f, 0.70f, 0.68f);
		}
	}

	// buildings, parks and houses
	Building bl[4];
	getBuildings(cx, cz, bl);
	uint32_t ws = hash2(cx + 77, cz - 31) | 1u;
	for (int k = 0; k < 4; k++)
		if (bl[k].type != B_NONE) addBuilding(&bl[k], &ws);

	size_t bytes = vn * sizeof(Vertex);
	Vertex* v = (Vertex*)linearAlloc(bytes);
	if (!v) return false;
	memcpy(v, scratch, bytes);
	GSPGPU_FlushDataCache(v, bytes);

	slot->cx = cx; slot->cz = cz; slot->used = true; slot->verts = v; slot->nverts = vn;
	loadedCount++;
	return true;
}

static void unloadChunk(Chunk* c)
{
	linearFree(c->verts);
	c->verts = NULL;
	c->used = false;
	loadedCount--;
}

// ---- streaming rules -------------------------------------------------------------------
static bool inKeepZone(float px, float pz, float margin, const StreamCtx* s)
{
	if (hypotf(px - s->carX, pz - s->carZ) < KEEP_CAR_R + margin) return true;
	float tx = px - s->camX, tz = pz - s->camZ;
	float dist = hypotf(tx, tz);
	float ahead = tx * s->dirX + tz * s->dirZ;
	float limit = (ahead >= 0.0f) ? s->keepAhead : KEEP_BEHIND;
	return dist < limit + margin;
}

static void updateStreaming(const StreamCtx* s, int maxLoads)
{
	for (int i = 0; i < MAX_CHUNKS; i++) {
		if (!chunks[i].used) continue;
		float px = (chunks[i].cx + 0.5f) * CHUNK_SIZE;
		float pz = (chunks[i].cz + 0.5f) * CHUNK_SIZE;
		if (!inKeepZone(px, pz, UNLOAD_MARGIN, s)) unloadChunk(&chunks[i]);
	}

	int ccx = (int)floorf(s->camX / CHUNK_SIZE);
	int ccz = (int)floorf(s->camZ / CHUNK_SIZE);
	for (int n = 0; n < maxLoads; n++) {
		int bx = 0, bz = 0; float bestD = 1e30f; bool found = false;
		for (int dz = -SCAN_RADIUS; dz <= SCAN_RADIUS; dz++) {
			for (int dx = -SCAN_RADIUS; dx <= SCAN_RADIUS; dx++) {
				int cx = ccx + dx, cz = ccz + dz;
				if (findChunk(cx, cz)) continue;
				float px = (cx + 0.5f) * CHUNK_SIZE, pz = (cz + 0.5f) * CHUNK_SIZE;
				if (!inKeepZone(px, pz, 0.0f, s)) continue;
				float d = hypotf(px - s->carX, pz - s->carZ);
				if (d < bestD) { bestD = d; bx = cx; bz = cz; found = true; }
			}
		}
		if (!found || !loadChunk(bx, bz)) break;
	}
}

// ---- car meshes (body + cabin) ------------------------------------------------------------
static void addBox(Vertex* out, int* n, float x0, float y0, float z0,
                   float x1, float y1, float z1, const float cols[6][3])
{
	// faces: 0 front(-z) 1 back(+z) 2 bottom 3 top 4 left(-x) 5 right(+x)
	static const int faces[6][4] = {
		{0,1,3,2}, {4,6,7,5}, {0,4,5,1}, {2,3,7,6}, {0,2,6,4}, {1,5,7,3}
	};
	for (int f = 0; f < 6; f++) {
		Vertex q[4];
		for (int k = 0; k < 4; k++) {
			int c = faces[f][k];
			q[k].x = (c & 1) ? x1 : x0;
			q[k].y = (c & 2) ? y1 : y0;
			q[k].z = (c & 4) ? z1 : z0;
			q[k].r = cols[f][0]; q[k].g = cols[f][1]; q[k].b = cols[f][2]; q[k].a = 1.0f;
		}
		out[(*n)++] = q[0]; out[(*n)++] = q[1]; out[(*n)++] = q[2];
		out[(*n)++] = q[0]; out[(*n)++] = q[2]; out[(*n)++] = q[3];
	}
}

#define CAR_MESH_VERTS 72
#define COP_MESH_VERTS 144

static Vertex* buildCarMesh(float r, float g, float b, bool yellowNose)
{
	Vertex* m = (Vertex*)linearAlloc(CAR_MESH_VERTS * sizeof(Vertex));
	int n = 0;
	float body[6][3] = {
		{ r * 0.95f, g * 0.95f, b * 0.95f }, { r * 0.55f, g * 0.55f, b * 0.55f },
		{ 0.15f, 0.15f, 0.15f }, { r, g, b },
		{ r * 0.8f, g * 0.8f, b * 0.8f }, { r * 0.8f, g * 0.8f, b * 0.8f }
	};
	if (yellowNose) { body[0][0] = 1.0f; body[0][1] = 0.85f; body[0][2] = 0.1f; }
	addBox(m, &n, -1.0f, 0.25f, -2.0f, 1.0f, 1.0f, 2.0f, body);

	float cabin[6][3] = {
		{ 0.12f, 0.16f, 0.22f }, { 0.12f, 0.16f, 0.22f }, { 0.1f, 0.1f, 0.1f },
		{ r * 0.9f, g * 0.9f, b * 0.9f }, { 0.12f, 0.16f, 0.22f }, { 0.12f, 0.16f, 0.22f }
	};
	addBox(m, &n, -0.85f, 1.0f, -0.7f, 0.85f, 1.65f, 1.2f, cabin);
	GSPGPU_FlushDataCache(m, CAR_MESH_VERTS * sizeof(Vertex));
	return m;
}

static void fillCols(float c[6][3], float r, float g, float b)
{
	for (int f = 0; f < 6; f++) { c[f][0] = r; c[f][1] = g; c[f][2] = b; }
}

// police car: white doors, dark roof, light bar that alternates red/blue by phase
static Vertex* buildPoliceMesh(int phase)
{
	Vertex* m = (Vertex*)linearAlloc(COP_MESH_VERTS * sizeof(Vertex));
	int n = 0;
	float body[6][3] = {
		{ 0.85f, 0.85f, 0.90f }, { 0.70f, 0.70f, 0.75f }, { 0.10f, 0.10f, 0.10f },
		{ 0.10f, 0.10f, 0.13f }, { 0.88f, 0.88f, 0.92f }, { 0.88f, 0.88f, 0.92f }
	};
	addBox(m, &n, -1.0f, 0.25f, -2.0f, 1.0f, 1.0f, 2.0f, body);
	float cabin[6][3] = {
		{ 0.12f, 0.16f, 0.22f }, { 0.12f, 0.16f, 0.22f }, { 0.1f, 0.1f, 0.1f },
		{ 0.08f, 0.08f, 0.10f }, { 0.12f, 0.16f, 0.22f }, { 0.12f, 0.16f, 0.22f }
	};
	addBox(m, &n, -0.85f, 1.0f, -0.7f, 0.85f, 1.65f, 1.2f, cabin);
	float red[6][3], blue[6][3];
	if (phase == 0) { fillCols(red, 1.0f, 0.15f, 0.15f); fillCols(blue, 0.12f, 0.15f, 0.35f); }
	else            { fillCols(red, 0.35f, 0.08f, 0.08f); fillCols(blue, 0.25f, 0.45f, 1.0f); }
	addBox(m, &n, -0.60f, 1.65f, -0.2f, -0.05f, 1.85f, 0.4f, red);
	addBox(m, &n, 0.05f, 1.65f, -0.2f, 0.60f, 1.85f, 0.4f, blue);
	GSPGPU_FlushDataCache(m, COP_MESH_VERTS * sizeof(Vertex));
	return m;
}

// ---- cars: collisions, civilians, police ----------------------------------------------------
// true if a point is inside a building, tunnel wall or the river (plus a margin)
static bool pointBlocked(float x, float z, float margin)
{
	int cx = (int)floorf(x / CHUNK_SIZE), cz = (int)floorf(z / CHUNK_SIZE);
	Building sol[MAX_SOLIDS];
	int n = getSolids(cx, cz, sol);
	for (int i = 0; i < n; i++)
		if (x > sol[i].x0 - margin && x < sol[i].x1 + margin &&
		    z > sol[i].z0 - margin && z < sol[i].z1 + margin) return true;
	return false;
}

// push a car out of solids; hitting them at speed damages it. Returns impact speed.
static float pushOutOfSolids(Car* c, float dt)
{
	int cx = (int)floorf(c->x / CHUNK_SIZE), cz = (int)floorf(c->z / CHUNK_SIZE);
	Building sol[MAX_SOLIDS];
	int ns = getSolids(cx, cz, sol);
	float hit = 0.0f;
	for (int b = 0; b < ns; b++) {
		float nx = clampf(c->x, sol[b].x0, sol[b].x1);
		float nz = clampf(c->z, sol[b].z0, sol[b].z1);
		float ex = c->x - nx, ez = c->z - nz;
		float d2 = ex * ex + ez * ez;
		if (d2 >= CAR_R * CAR_R) continue;
		if (d2 > 1e-6f) {
			float d = sqrtf(d2), p = (CAR_R - d) / d;
			c->x += ex * p; c->z += ez * p;
		} else {
			float l = c->x - sol[b].x0, r = sol[b].x1 - c->x;
			float u = c->z - sol[b].z0, dn = sol[b].z1 - c->z;
			float m = fminf(fminf(l, r), fminf(u, dn));
			if (m == l)      c->x = sol[b].x0 - CAR_R;
			else if (m == r) c->x = sol[b].x1 + CAR_R;
			else if (m == u) c->z = sol[b].z0 - CAR_R;
			else             c->z = sol[b].z1 + CAR_R;
		}
		hit = fmaxf(hit, fabsf(c->speed));
		c->vx *= 0.5f; c->vz *= 0.5f;
	}
	if (hit > 0.0f) {
		if (hit > 5.0f) c->hp -= (hit - 5.0f) * dt * 3.0f;
		c->speed *= expf(-5.0f * dt);
	}
	return hit;
}

// apply an impulse: the forward part changes speed, the sideways part becomes a slide,
// and an off-center hit adds yaw spin. (cnx,cnz) points from the car toward the contact.
static void applyImpulse(Car* c, float dvx, float dvz, float cnx, float cnz)
{
	float fx = sinf(c->heading), fz = -cosf(c->heading);
	float rx = cosf(c->heading), rz = sinf(c->heading);
	float dF = dvx * fx + dvz * fz, dL = dvx * rx + dvz * rz;
	c->speed += dF;
	c->vx += rx * dL; c->vz += rz * dL;
	c->spin += 0.06f * ((cnx * fx + cnz * fz) * CAR_R) * dL;
}

// circle-circle car collision with momentum exchange and damage
static bool carCollide(Car* a, Car* b, float* impulse)
{
	*impulse = 0.0f;
	float dx = b->x - a->x, dz = b->z - a->z;
	float d2 = dx * dx + dz * dz, minD = 2.0f * CAR_R;
	if (d2 >= minD * minD) return false;
	float d = sqrtf(d2), nx = 1.0f, nz = 0.0f;
	if (d > 0.01f) { nx = dx / d; nz = dz / d; } else d = 0.01f;
	float pen = minD - d;
	a->x -= nx * pen * 0.5f; a->z -= nz * pen * 0.5f;
	b->x += nx * pen * 0.5f; b->z += nz * pen * 0.5f;

	float avx = sinf(a->heading) * a->speed + a->vx, avz = -cosf(a->heading) * a->speed + a->vz;
	float bvx = sinf(b->heading) * b->speed + b->vx, bvz = -cosf(b->heading) * b->speed + b->vz;
	float rel = (bvx - avx) * nx + (bvz - avz) * nz;
	if (rel >= 0.0f) return true;          // already separating

	float j = -1.25f * rel * 0.5f;         // equal masses, restitution 0.25
	applyImpulse(a, -j * nx, -j * nz,  nx,  nz);
	applyImpulse(b,  j * nx,  j * nz, -nx, -nz);
	float dmg = fmaxf(0.0f, j - 1.5f) * 2.4f;
	a->hp -= dmg; b->hp -= dmg;
	*impulse = j;
	return true;
}

// is the point (ox,oz) in front of lane-driving car t?
static bool aheadOf(const Car* t, float ox, float oz)
{
	float fx = dxs[t->dir], fz = dzs[t->dir];
	float rx = ox - t->x, rz = oz - t->z;
	float ahead = rx * fx + rz * fz;
	float lat = fabsf(-rx * fz + rz * fx);
	return ahead > 0.0f && ahead < 14.0f && lat < 2.6f;
}

static void spawnCivilian(float px, float pz)
{
	int civs = 0;
	Car* c = NULL;
	for (int i = 0; i < MAX_CARS; i++) {
		if (!cars[i].active) { if (!c) c = &cars[i]; }
		else if (cars[i].kind == K_CIV) civs++;
	}
	if (!c || civs >= MAX_CIV) return;

	for (int attempt = 0; attempt < 4; attempt++) {
		float ang = frand() * 2.0f * PI, d = 50.0f + frand() * 35.0f;
		float x = px + cosf(ang) * d, z = pz + sinf(ang) * d;
		int dir = (int)(frand() * 4.0f) & 3;
		float lane;
		if (dir & 1) {
			float line = roundf(z / CHUNK_SIZE) * CHUNK_SIZE;
			z = line + laneOff[dir]; lane = z;
		} else {
			int li = (int)roundf(x / CHUNK_SIZE);
			if (isFreewayX(li)) continue;          // no street traffic on freeway lines
			x = li * CHUNK_SIZE + laneOff[dir]; lane = x;
		}

		float dd = hypotf(x - px, z - pz);
		if (dd < 45.0f || dd > 90.0f) continue;
		bool ok = true;
		for (int k = 0; k < MAX_CARS; k++)
			if (cars[k].active && hypotf(cars[k].x - x, cars[k].z - z) < 10.0f) { ok = false; break; }
		if (!ok) continue;

		memset(c, 0, sizeof *c);
		c->active = true; c->kind = K_CIV;
		c->x = x; c->z = z; c->dir = dir; c->lane = lane;
		c->heading = dirHeading[dir];
		c->cruise = 8.0f + frand() * 6.0f; c->speed = c->cruise;
		c->hp = 100.0f;
		c->color = 1 + ((int)(frand() * 6.0f) % 6);
		return;
	}
}

static void spawnCop(float px, float pz)
{
	Car* c = NULL;
	for (int i = 0; i < MAX_CARS; i++)
		if (!cars[i].active) { c = &cars[i]; break; }
	if (!c) return;

	for (int attempt = 0; attempt < 6; attempt++) {
		float ang = frand() * 2.0f * PI, d = 90.0f + frand() * 40.0f;
		float x = px + cosf(ang) * d, z = pz + sinf(ang) * d;
		int dir = (int)(frand() * 4.0f) & 3;
		if (dir & 1) {
			z = roundf(z / CHUNK_SIZE) * CHUNK_SIZE + laneOff[dir];
		} else {
			int li = (int)roundf(x / CHUNK_SIZE);
			if (isFreewayX(li)) continue;
			x = li * CHUNK_SIZE + laneOff[dir];
		}
		float dd = hypotf(x - px, z - pz);
		if (dd < 70.0f || dd > 140.0f) continue;
		if (pointBlocked(x, z, 3.0f)) continue;
		bool ok = true;
		for (int k = 0; k < MAX_CARS; k++)
			if (cars[k].active && hypotf(cars[k].x - x, cars[k].z - z) < 10.0f) { ok = false; break; }
		if (!ok) continue;

		memset(c, 0, sizeof *c);
		c->active = true; c->kind = K_COP;
		c->x = x; c->z = z;
		c->heading = atan2f(px - x, -(pz - z));   // facing the player
		c->speed = 18.0f; c->hp = 100.0f;
		return;
	}
}

// civilian driving in its lane, stopping for anything ahead, turning randomly at intersections
static void civLane(Car* t, float dt, int self, bool playerUp)
{
	float tgt = t->cruise;
	if (!playerUp && aheadOf(t, player.x, player.z)) tgt = 0.0f;
	for (int k = 0; k < MAX_CARS; k++)
		if (k != self && cars[k].active && aheadOf(t, cars[k].x, cars[k].z)) tgt = 0.0f;

	if (t->speed > tgt) { t->speed -= 18.0f * dt; if (t->speed < tgt) t->speed = tgt; }
	else                { t->speed += 6.0f * dt;  if (t->speed > tgt) t->speed = tgt; }

	t->stuck = (t->speed < 0.5f) ? t->stuck + dt : 0.0f;
	if (t->stuck > 5.0f) { t->active = false; return; }   // break up jams

	float prevAlong = (t->dir & 1) ? t->x : t->z;
	t->x += dxs[t->dir] * t->speed * dt;
	t->z += dzs[t->dir] * t->speed * dt;
	float lk = 1.0f - expf(-3.0f * dt);
	if (t->dir & 1) t->z += (t->lane - t->z) * lk;
	else            t->x += (t->lane - t->x) * lk;
	float along = (t->dir & 1) ? t->x : t->z;

	// crossed an intersection center line? maybe turn
	if (floorf(prevAlong / CHUNK_SIZE) != floorf(along / CHUNK_SIZE)) {
		float crossed = fmaxf(floorf(prevAlong / CHUNK_SIZE), floorf(along / CHUNK_SIZE)) * CHUNK_SIZE;
		float r = frand();
		int nd = t->dir;
		if (r < 0.18f) nd = (t->dir + 1) & 3;        // right
		else if (r < 0.36f) nd = (t->dir + 3) & 3;   // left
		// never turn onto a freeway line
		if ((nd & 1) == 0 && nd != t->dir && isFreewayX((int)roundf(crossed / CHUNK_SIZE))) nd = t->dir;
		if (nd != t->dir) { t->dir = nd; t->lane = crossed + laneOff[nd]; }
	}

	t->heading += angleDiff(t->heading, dirHeading[t->dir]) * (1.0f - expf(-6.0f * dt));
}

// police pursuit: steer toward the player (with a lead), avoid buildings with feelers,
// ram at higher levels, and give up when the heat is gone
static void copAI(Car* c, float dt)
{
	if (heat <= 0.0f) {                       // retiring
		c->lostT += dt;
		c->speed *= expf(-0.6f * dt);
		if (c->lostT > 5.0f) c->active = false;
		return;
	}
	c->lostT = 0.0f;
	int level = (int)ceilf(heat / 20.0f);
	if (level < 1) level = 1;
	if (level > 5) level = 5;

	float tx = player.x + (sinf(player.heading) * player.speed + player.vx) * 0.4f;
	float tz = player.z + (-cosf(player.heading) * player.speed + player.vz) * 0.4f;
	float dx = tx - c->x, dz = tz - c->z;
	float dist = hypotf(dx, dz);
	float want = atan2f(dx, -dz);

	// feelers: pick the first direction (straight at the player, then to either side) that is clear
	static const float offs[7] = { 0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1.6f, -1.6f };
	float look = 10.0f + fabsf(c->speed) * 0.6f;
	float chosen = want;
	bool found = false;
	for (int k = 0; k < 7; k++) {
		float a = want + offs[k];
		float fx = sinf(a), fz = -cosf(a);
		if (!pointBlocked(c->x + fx * look, c->z + fz * look, 2.0f) &&
		    !pointBlocked(c->x + fx * look * 0.5f, c->z + fz * look * 0.5f, 2.0f)) {
			chosen = a; found = true; break;
		}
	}

	float diff = angleDiff(c->heading, chosen);
	float turn = 2.6f * clampf(fabsf(c->speed) / 10.0f, 0.2f, 1.0f);
	c->heading += clampf(diff, -turn * dt, turn * dt);

	float tgt = 30.0f + level * 3.0f;                    // 33..45 m/s
	if (fabsf(diff) > 0.9f) tgt *= 0.45f;                // sharp turn: slow down
	if (dist < 14.0f) {
		float ps = fabsf(player.speed);
		tgt = (level >= 3) ? ps + 10.0f : fmaxf(8.0f, ps);   // ram at 3+ stars, otherwise shadow
	}
	if (!found) tgt = fminf(tgt, 6.0f);
	if (c->speed < tgt) c->speed += 20.0f * dt;
	else                c->speed -= 28.0f * dt;
}

// advance all cars one step, resolve car-car collisions (including the player), count cops
static void updateCars(float dt, float py)
{
	bool playerUp = py > 2.0f;
	int copCount = 0, civCount = 0;
	gNearestCop = 1e9f;

	for (int i = 0; i < MAX_CARS; i++) {
		Car* c = &cars[i];
		if (!c->active) continue;
		float dpl = hypotf(c->x - player.x, c->z - player.z);
		if (dpl > (c->kind == K_COP ? 140.0f : 96.0f)) { c->active = false; continue; }

		bool wrecked = c->hp <= 0.0f;
		if (wrecked && c->wreckT <= 0.0f) {
			c->wreckT = 0.001f; c->lost = true;
			if (c->kind == K_COP) { heat = fminf(100.0f, heat + 12.0f); copWrecks++; }
		}
		if (wrecked) {
			c->wreckT += dt;
			if (c->wreckT > 10.0f) { c->active = false; continue; }
			c->speed *= expf(-1.5f * dt);
		}

		if (!wrecked && c->kind == K_CIV && !c->lost) {
			civLane(c, dt, i, playerUp);
		} else {
			if (!wrecked) {
				if (c->kind == K_CIV) {            // knocked out of its lane: roll to a stop
					c->speed *= expf(-0.7f * dt);
					c->lostT += dt;
					if (c->lostT > 12.0f) { c->active = false; continue; }
				} else {
					copAI(c, dt);
				}
			}
			if (!c->active) continue;
			float fx = sinf(c->heading), fz = -cosf(c->heading);
			c->x += (fx * c->speed + c->vx) * dt;
			c->z += (fz * c->speed + c->vz) * dt;
			c->heading += c->spin * dt;
			c->vx *= expf(-2.5f * dt); c->vz *= expf(-2.5f * dt);
			c->spin *= expf(-3.0f * dt);
			pushOutOfSolids(c, dt);
		}

		if (!c->active) continue;
		if (c->kind == K_COP) {
			if (!wrecked) { copCount++; gNearestCop = fminf(gNearestCop, dpl); }
		} else {
			civCount++;
		}
	}

	// collisions: car vs car, and player vs car (only when the player is at street level)
	for (int i = 0; i < MAX_CARS; i++) {
		if (!cars[i].active) continue;
		for (int k = i + 1; k < MAX_CARS; k++) {
			if (!cars[k].active) continue;
			float j;
			if (carCollide(&cars[i], &cars[k], &j) && j > 1.0f) {
				if (cars[i].kind == K_CIV) cars[i].lost = true;
				if (cars[k].kind == K_CIV) cars[k].lost = true;
			}
		}
		if (!playerUp) {
			float j;
			if (carCollide(&player, &cars[i], &j) && j > 1.0f) {
				if (cars[i].kind == K_CIV) cars[i].lost = true;
				if (j > 3.0f && hitCool <= 0.0f) {
					heat = fminf(100.0f, heat + (cars[i].kind == K_CIV ? 18.0f : 8.0f));
					hitCool = 0.8f;
				}
			}
		}
	}
	gCivCount = civCount; gCopCount = copCount;
}

// ---- minimap (bottom screen, drawn in 2D with the same shader) ---------------------------------
#define MAP_MAX_VERTS 12000

static Vertex* mapBuf[2];
static Vertex* mapOut;
static int mapN;
static float mcx, mcy, mscale, mcos, msin, mCarX, mCarZ;

static const float C_BLOCK[3]   = { 0.105f, 0.130f, 0.180f };
static const float C_ROAD[3]    = { 0.360f, 0.400f, 0.480f };
static const float C_AVE[3]     = { 0.450f, 0.490f, 0.570f };
static const float C_LINE[3]    = { 0.850f, 0.720f, 0.280f };
static const float C_MEDIAN[3]  = { 0.200f, 0.420f, 0.230f };
static const float C_WATER[3]   = { 0.100f, 0.260f, 0.440f };
static const float C_FWY_E[3]   = { 0.960f, 0.560f, 0.200f };
static const float C_FWY_G[3]   = { 0.560f, 0.340f, 0.200f };
static const float C_FWY_R[3]   = { 0.800f, 0.500f, 0.240f };
static const float C_TUNNEL[3]  = { 0.060f, 0.075f, 0.120f };
static const float C_TUNEDGE[3] = { 0.500f, 0.560f, 0.680f };
static const float C_FRAME[3]   = { 0.040f, 0.050f, 0.075f };
static const float C_FRAME2[3]  = { 0.330f, 0.380f, 0.480f };
static const float C_PANEL[3]   = { 0.060f, 0.075f, 0.110f };
static const float C_WHITE[3]   = { 0.950f, 0.960f, 1.000f };
static const float C_YELLOW[3]  = { 1.000f, 0.820f, 0.150f };
static const float C_RED[3]     = { 0.950f, 0.220f, 0.200f };
static const float C_GRAY[3]    = { 0.550f, 0.580f, 0.650f };
static const float C_COP_R[3]   = { 1.000f, 0.200f, 0.200f };
static const float C_COP_B[3]   = { 0.250f, 0.480f, 1.000f };

static void mTri(float x0, float y0, float x1, float y1, float x2, float y2, const float* c)
{
	if (mapN + 3 > MAP_MAX_VERTS) return;
	float p[6] = { x0, y0, x1, y1, x2, y2 };
	for (int i = 0; i < 3; i++) {
		Vertex* v = &mapOut[mapN++];
		v->x = p[i * 2]; v->y = p[i * 2 + 1]; v->z = 0.0f;
		v->r = c[0]; v->g = c[1]; v->b = c[2]; v->a = 1.0f;
	}
}

static void mQuad(float x0, float y0, float x1, float y1, float x2, float y2,
                  float x3, float y3, const float* c)
{
	mTri(x0, y0, x1, y1, x2, y2, c);
	mTri(x0, y0, x2, y2, x3, y3, c);
}

static void mRect(float x, float y, float w, float h, const float* c)
{
	mQuad(x, y, x + w, y, x + w, y + h, x, y + h, c);
}

// world (meters) -> screen (pixels), rotated by the map angle
static void w2s(float wx, float wz, float* sx, float* sy)
{
	float dx = wx - mCarX, dz = wz - mCarZ;
	*sx = mcx + mscale * (dx * mcos + dz * msin);
	*sy = mcy - mscale * (dx * msin - dz * mcos);
}

static void mWorldQuad(float x0, float z0, float x1, float z1,
                       float x2, float z2, float x3, float z3, const float* c)
{
	float p[8];
	w2s(x0, z0, &p[0], &p[1]); w2s(x1, z1, &p[2], &p[3]);
	w2s(x2, z2, &p[4], &p[5]); w2s(x3, z3, &p[6], &p[7]);
	mQuad(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], c);
}

static void mWorldRect(float xa, float za, float xb, float zb, const float* c)
{
	mWorldQuad(xa, za, xb, za, xb, zb, xa, zb, c);
}

static void mWorldCar(float x, float z, float h, float hw, float hl, const float* c)
{
	float fx = sinf(h), fz = -cosf(h), rx = cosf(h), rz = sinf(h);
	mWorldQuad(x - rx * hw - fx * hl, z - rz * hw - fz * hl,
	           x + rx * hw - fx * hl, z + rz * hw - fz * hl,
	           x + rx * hw + fx * hl, z + rz * hw + fz * hl,
	           x - rx * hw + fx * hl, z - rz * hw + fz * hl, c);
}

static void mDisc(float cx, float cy, float r, const float* c)
{
	for (int i = 0; i < 20; i++) {
		float a0 = i * 2.0f * PI / 20.0f, a1 = (i + 1) * 2.0f * PI / 20.0f;
		mTri(cx, cy, cx + cosf(a0) * r, cy + sinf(a0) * r, cx + cosf(a1) * r, cy + sinf(a1) * r, c);
	}
}

static void mDiamond(float cx, float cy, float r, const float* c)
{
	mTri(cx, cy - r, cx + r, cy, cx, cy + r, c);
	mTri(cx, cy - r, cx, cy + r, cx - r, cy, c);
}

// player arrow, ang = clockwise rotation from "up" in radians
static void mChevron(float cx, float cy, float ang, float s, const float* c)
{
	static const float L[4][2] = { { 0, -11 }, { -7, 8 }, { 0, 3 }, { 7, 8 } };
	float ca = cosf(ang), sa = sinf(ang), p[4][2];
	for (int i = 0; i < 4; i++) {
		float x = L[i][0] * s, y = L[i][1] * s;
		p[i][0] = cx + x * ca - y * sa;
		p[i][1] = cy + x * sa + y * ca;
	}
	mTri(p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], c);
	mTri(p[0][0], p[0][1], p[2][0], p[2][1], p[3][0], p[3][1], c);
}

// tiny 3x5 pixel font: digits and the letters the HUD needs
typedef struct { char c; unsigned char r[5]; } Glyph;
static const Glyph glyphs[] = {
	{ '0', { 7, 5, 5, 5, 7 } }, { '1', { 2, 6, 2, 2, 7 } }, { '2', { 7, 1, 7, 4, 7 } },
	{ '3', { 7, 1, 7, 1, 7 } }, { '4', { 5, 5, 7, 1, 1 } }, { '5', { 7, 4, 7, 1, 7 } },
	{ '6', { 7, 4, 7, 5, 7 } }, { '7', { 7, 1, 2, 2, 2 } }, { '8', { 7, 5, 7, 5, 7 } },
	{ '9', { 7, 5, 7, 1, 7 } }, { 'K', { 5, 5, 6, 5, 5 } }, { 'M', { 5, 7, 7, 5, 5 } },
	{ 'H', { 5, 5, 7, 5, 5 } }, { 'N', { 7, 5, 5, 5, 5 } }, { '/', { 1, 1, 2, 4, 4 } },
	{ 'A', { 2, 5, 7, 5, 5 } }, { 'B', { 6, 5, 6, 5, 6 } }, { 'C', { 3, 4, 4, 4, 3 } },
	{ 'D', { 6, 5, 5, 5, 6 } }, { 'E', { 7, 4, 6, 4, 7 } }, { 'I', { 7, 2, 2, 2, 7 } },
	{ 'L', { 4, 4, 4, 4, 7 } }, { 'O', { 2, 5, 5, 5, 2 } }, { 'P', { 6, 5, 6, 4, 4 } },
	{ 'R', { 6, 5, 6, 5, 5 } }, { 'S', { 3, 4, 2, 1, 6 } }, { 'T', { 7, 2, 2, 2, 2 } },
	{ 'U', { 5, 5, 5, 5, 7 } }, { 'W', { 5, 5, 7, 7, 5 } }, { 'Y', { 5, 5, 2, 2, 2 } }
};

static void mText(const char* str, float x, float y, float px, const float* c)
{
	for (; *str; str++) {
		for (unsigned gi = 0; gi < sizeof(glyphs) / sizeof(glyphs[0]); gi++) {
			if (glyphs[gi].c != *str) continue;
			for (int row = 0; row < 5; row++)
				for (int col = 0; col < 3; col++)
					if ((glyphs[gi].r[row] >> (2 - col)) & 1)
						mRect(x + col * px, y + row * px, px, px, c);
			break;
		}
		x += 4.0f * px;
	}
}

static void buildMinimap(Vertex* out, float px, float pz, float heading, float speed,
                         bool headingUp, float scale)
{
	mapOut = out; mapN = 0;
	float a = headingUp ? heading : 0.0f;
	mcx = 160.0f; mcy = headingUp ? 172.0f : 120.0f;
	mscale = scale; mcos = cosf(a); msin = sinf(a); mCarX = px; mCarZ = pz;

	const float S = CHUNK_SIZE;
	float R = 245.0f / scale; // world radius that covers the whole screen
	int x0 = (int)floorf((px - R) / S), x1 = (int)floorf((px + R) / S);
	int z0 = (int)floorf((pz - R) / S), z1 = (int)floorf((pz + R) / S);

	// city blocks and rivers
	for (int cz = z0; cz <= z1; cz++) {
		for (int cx = x0; cx <= x1; cx++) {
			mWorldRect(cx * S, cz * S, (cx + 1) * S, (cz + 1) * S, C_BLOCK);
			if (isRiverRow(cz))
				mWorldRect(cx * S, cz * S + roadHalfZ(cz) + 2.0f, (cx + 1) * S,
				           (cz + 1) * S - roadHalfZ(cz + 1) - 2.0f, C_WATER);
		}
	}

	// roads (freeways are drawn separately); avenues are wider and brighter
	for (int gx = x0; gx <= x1 + 1; gx++) {
		if (isFreewayX(gx)) continue;
		float hw = roadHalfX(gx);
		mWorldRect(gx * S - hw, z0 * S, gx * S + hw, (z1 + 1) * S, isAvenueX(gx) ? C_AVE : C_ROAD);
	}
	for (int gz = z0; gz <= z1 + 1; gz++) {
		float hw = roadHalfZ(gz);
		mWorldRect(x0 * S, gz * S - hw, (x1 + 1) * S, gz * S + hw, isAvenueZ(gz) ? C_AVE : C_ROAD);
	}

	// center lines (amber on streets, green median on avenues), none inside intersections
	for (int gx = x0; gx <= x1 + 1; gx++) {
		if (isFreewayX(gx)) continue;
		bool ave = isAvenueX(gx);
		for (int cz = z0; cz <= z1; cz++) {
			if (tunnelOnLine(gx, cz)) continue;
			mWorldRect(gx * S - (ave ? 0.9f : 0.5f), cz * S + roadHalfZ(cz),
			           gx * S + (ave ? 0.9f : 0.5f), (cz + 1) * S - roadHalfZ(cz + 1),
			           ave ? C_MEDIAN : C_LINE);
		}
	}
	for (int gz = z0; gz <= z1 + 1; gz++) {
		bool ave = isAvenueZ(gz);
		for (int cx = x0; cx <= x1; cx++)
			mWorldRect(cx * S + roadHalfX(cx), gz * S - (ave ? 0.9f : 0.5f),
			           (cx + 1) * S - roadHalfX(cx + 1), gz * S + (ave ? 0.9f : 0.5f),
			           ave ? C_MEDIAN : C_LINE);
	}

	// tunnels: dark road with light edges
	for (int gx = x0; gx <= x1 + 1; gx++) {
		for (int cz = z0; cz <= z1; cz++) {
			if (!tunnelOnLine(gx, cz)) continue;
			float zs = cz * S + TUN_END, ze = (cz + 1) * S - TUN_END;
			mWorldRect(gx * S - ROAD_HALF, zs, gx * S + ROAD_HALF, ze, C_TUNNEL);
			mWorldRect(gx * S - ROAD_HALF - 0.6f, zs, gx * S - ROAD_HALF + 0.6f, ze, C_TUNEDGE);
			mWorldRect(gx * S + ROAD_HALF - 0.6f, zs, gx * S + ROAD_HALF + 0.6f, ze, C_TUNEDGE);
		}
	}

	// freeways: elevated = bright, ramp = mid, ground = dim
	for (int gx = x0; gx <= x1 + 1; gx++) {
		if (!isFreewayX(gx)) continue;
		float lx = gx * S;
		for (int cz = z0; cz <= z1; cz++) {
			float fa = fwyCrossH(gx, cz), fb = fwyCrossH(gx, cz + 1);
			float zA = cz * S;
			const float* cA = fa > 0.5f ? C_FWY_E : C_FWY_G;
			const float* cB = fb > 0.5f ? C_FWY_E : C_FWY_G;
			const float* cM = (fa > 0.5f && fb > 0.5f) ? C_FWY_E
			                  : ((fa < 0.5f && fb < 0.5f) ? C_FWY_G : C_FWY_R);
			mWorldRect(lx - FWY_HALF, zA, lx + FWY_HALF, zA + 10.0f, cA);
			mWorldRect(lx - FWY_HALF, zA + 10.0f, lx + FWY_HALF, zA + 54.0f, cM);
			mWorldRect(lx - FWY_HALF, zA + 54.0f, lx + FWY_HALF, zA + 64.0f, cB);
			mWorldRect(lx - 0.5f, zA, lx + 0.5f, zA + 64.0f, C_TUNNEL);
		}
	}

	// buildings, colored by type (towers brighter and bluer when taller)
	for (int cz = z0; cz <= z1; cz++) {
		for (int cx = x0; cx <= x1; cx++) {
			Building bl[4];
			getBuildings(cx, cz, bl);
			for (int i = 0; i < 4; i++) {
				if (bl[i].type == B_NONE) continue;
				float t = clampf(bl[i].h / 60.0f, 0.0f, 1.0f);
				float c[3];
				if (bl[i].type == B_PARK)       { c[0] = 0.15f; c[1] = 0.36f; c[2] = 0.19f; }
				else if (bl[i].type == B_HOUSE) { c[0] = 0.33f; c[1] = 0.27f; c[2] = 0.26f; }
				else if (bl[i].type == B_WARE)  { c[0] = 0.22f; c[1] = 0.23f; c[2] = 0.27f; }
				else if (bl[i].type == B_TOWER) { c[0] = 0.18f + 0.18f * t; c[1] = 0.30f + 0.22f * t; c[2] = 0.50f + 0.25f * t; }
				else { c[0] = 0.16f + 0.14f * t; c[1] = 0.20f + 0.16f * t; c[2] = 0.29f + 0.18f * t; }
				mWorldRect(bl[i].x0, bl[i].z0, bl[i].x1, bl[i].z1, c);
			}
		}
	}

	// cars: police always (flashing), civilians only if enabled
	for (int i = 0; i < MAX_CARS; i++) {
		if (!cars[i].active) continue;
		if (cars[i].kind == K_COP) {
			if (cars[i].hp <= 0.0f) continue;
			mWorldCar(cars[i].x, cars[i].z, cars[i].heading, 2.3f, 4.2f, C_WHITE);
			mWorldCar(cars[i].x, cars[i].z, cars[i].heading, 1.7f, 3.6f, gCopPhase ? C_COP_B : C_COP_R);
		}
#if SHOW_TRAFFIC_ON_MAP
		else {
			mWorldCar(cars[i].x, cars[i].z, cars[i].heading, 1.6f, 3.2f, trafficPal[cars[i].color - 1]);
		}
#endif
	}

	// player arrow (always drawn at the map center)
	mChevron(mcx, mcy, heading - a, 1.4f, C_WHITE);
	mChevron(mcx, mcy, heading - a, 1.0f, C_YELLOW);

	// compass dial: the red marker always points north
	{
		float cx = 288.0f, cy = 36.0f;
		float nx = -msin, ny = -mcos;   // screen direction of north
		float tx = -ny, ty = nx;
		mDisc(cx, cy, 25.0f, C_FRAME2);
		mDisc(cx, cy, 23.0f, C_PANEL);
		mTri(cx + nx * 22.0f, cy + ny * 22.0f,
		     cx + nx * 14.0f + tx * 5.0f, cy + ny * 14.0f + ty * 5.0f,
		     cx + nx * 14.0f - tx * 5.0f, cy + ny * 14.0f - ty * 5.0f, C_RED);
		mTri(cx - nx * 20.0f, cy - ny * 20.0f,
		     cx - nx * 14.0f + tx * 4.0f, cy - ny * 14.0f + ty * 4.0f,
		     cx - nx * 14.0f - tx * 4.0f, cy - ny * 14.0f - ty * 4.0f, C_GRAY);
		mText("N", cx + nx * 6.0f - 2.25f, cy + ny * 6.0f - 3.75f, 1.5f, C_WHITE);
	}

	// speed panel
	{
		char buf[8];
		snprintf(buf, sizeof buf, "%d", (int)(fabsf(speed) * 3.6f + 0.5f));
		mRect(8.0f, 194.0f, 118.0f, 40.0f, C_FRAME2);
		mRect(9.0f, 195.0f, 116.0f, 38.0f, C_PANEL);
		mText(buf, 18.0f, 201.0f, 5.0f, C_WHITE);
		mText("KM/H", 84.0f, 218.0f, 2.0f, C_GRAY);
	}

	// health panel (bottom right)
	{
		float hpF = clampf(player.hp / 100.0f, 0.0f, 1.0f);
		mRect(194.0f, 194.0f, 118.0f, 40.0f, C_FRAME2);
		mRect(195.0f, 195.0f, 116.0f, 38.0f, C_PANEL);
		if (player.hp > 0.0f) {
			float bc[3] = { 0.85f - 0.6f * hpF, 0.25f + 0.6f * hpF, 0.20f };
			mText("HEALTH", 201.0f, 200.0f, 2.0f, C_GRAY);
			mRect(201.0f, 214.0f, 104.0f, 12.0f, C_FRAME);
			mRect(201.0f, 214.0f, 104.0f * hpF, 12.0f, bc);
		} else {
			mText("WRECKED", 200.0f, 202.0f, 2.6f, C_RED);
			mText("SELECT REPAIR", 201.0f, 220.0f, 1.5f, C_WHITE);
		}
	}

	// heat panel (top left): stars flash red/blue while wanted, wreck counter on the right
	{
		char buf[8];
		int level = heat > 0.0f ? (int)ceilf(heat / 20.0f) : 0;
		mRect(8.0f, 8.0f, 110.0f, 44.0f, C_FRAME2);
		mRect(9.0f, 9.0f, 108.0f, 42.0f, C_PANEL);
		mText("HEAT", 15.0f, 14.0f, 2.0f, C_GRAY);
		snprintf(buf, sizeof buf, "W%d", copWrecks);
		mText(buf, 80.0f, 14.0f, 2.0f, C_WHITE);
		for (int i = 0; i < 5; i++) {
			const float* sc = (i < level) ? (gCopPhase ? C_COP_B : C_COP_R) : C_FRAME2;
			mDiamond(25.0f + i * 19.0f, 36.0f, 7.0f, sc);
		}
		mRect(15.0f, 46.0f, 96.0f, 3.0f, C_FRAME);
		mRect(15.0f, 46.0f, 96.0f * clampf(heat / 100.0f, 0.0f, 1.0f), 3.0f, C_YELLOW);
	}

	// busted banner
	if (bustedT > 0.0f) {
		mRect(60.0f, 92.0f, 200.0f, 56.0f, C_FRAME2);
		mRect(62.0f, 94.0f, 196.0f, 52.0f, C_PANEL);
		mText("BUSTED", 91.0f, 106.0f, 6.0f, C_RED);
	}

	// screen frame
	mRect(0.0f, 0.0f, 320.0f, 3.0f, C_FRAME);
	mRect(0.0f, 237.0f, 320.0f, 3.0f, C_FRAME);
	mRect(0.0f, 0.0f, 3.0f, 240.0f, C_FRAME);
	mRect(317.0f, 0.0f, 3.0f, 240.0f, C_FRAME);
	mRect(3.0f, 3.0f, 314.0f, 1.0f, C_FRAME2);
	mRect(3.0f, 236.0f, 314.0f, 1.0f, C_FRAME2);
	mRect(3.0f, 3.0f, 1.0f, 234.0f, C_FRAME2);
	mRect(316.0f, 3.0f, 1.0f, 234.0f, C_FRAME2);
}

// ---- main ------------------------------------------------------------------------------------
static C3D_Mtx projection, view, mapProj;
static shaderProgram_s program;
static int uLoc_projection, uLoc_modelView;

static void bindVerts(Vertex* v)
{
	C3D_BufInfo* bi = C3D_GetBufInfo();
	BufInfo_Init(bi);
	BufInfo_Add(bi, v, sizeof(Vertex), 2, 0x10);
}

// the player is a Car too, so it shares the collision code; these keep the main loop readable
#define carX       (player.x)
#define carZ       (player.z)
#define carHeading (player.heading)
#define carSpeed   (player.speed)

int main(void)
{
	gfxInitDefault();
	gfxSet3D(false);
	C3D_Init(C3D_DEFAULT_CMDBUF_SIZE);

	C3D_RenderTarget* top = C3D_RenderTargetCreate(240, 400, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
	C3D_RenderTargetSetOutput(top, GFX_TOP, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);
	C3D_RenderTarget* bot = C3D_RenderTargetCreate(240, 320, GPU_RB_RGBA8, GPU_RB_DEPTH24_STENCIL8);
	C3D_RenderTargetSetOutput(bot, GFX_BOTTOM, GFX_LEFT, DISPLAY_TRANSFER_FLAGS);

	DVLB_s* dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
	shaderProgramInit(&program);
	shaderProgramSetVsh(&program, &dvlb->DVLE[0]);
	C3D_BindProgram(&program);
	uLoc_projection = shaderInstanceGetUniformLocation(program.vertexShader, "projection");
	uLoc_modelView  = shaderInstanceGetUniformLocation(program.vertexShader, "modelView");

	C3D_AttrInfo* ai = C3D_GetAttrInfo();
	AttrInfo_Init(ai);
	AttrInfo_AddLoader(ai, 0, GPU_FLOAT, 3);
	AttrInfo_AddLoader(ai, 1, GPU_FLOAT, 4);

	C3D_TexEnv* env = C3D_GetTexEnv(0);
	C3D_TexEnvInit(env);
	C3D_TexEnvSrc(env, C3D_Both, GPU_PRIMARY_COLOR, 0, 0);
	C3D_TexEnvFunc(env, C3D_Both, GPU_REPLACE);

	C3D_CullFace(GPU_CULL_NONE);

	// car meshes: 0 = player, 1..6 = civilians, 7 = wreck, 8/9 = police light phases
	carMesh[0] = buildCarMesh(0.80f, 0.08f, 0.08f, true);
	carMeshN[0] = CAR_MESH_VERTS;
	for (int i = 0; i < 6; i++) {
		carMesh[i + 1] = buildCarMesh(trafficPal[i][0], trafficPal[i][1], trafficPal[i][2], false);
		carMeshN[i + 1] = CAR_MESH_VERTS;
	}
	carMesh[7] = buildCarMesh(0.13f, 0.13f, 0.14f, false);
	carMeshN[7] = CAR_MESH_VERTS;
	carMesh[8] = buildPoliceMesh(0); carMeshN[8] = COP_MESH_VERTS;
	carMesh[9] = buildPoliceMesh(1); carMeshN[9] = COP_MESH_VERTS;

	// two minimap vertex buffers (alternate each frame so the GPU never reads one being written)
	mapBuf[0] = (Vertex*)linearAlloc(MAP_MAX_VERTS * sizeof(Vertex));
	mapBuf[1] = (Vertex*)linearAlloc(MAP_MAX_VERTS * sizeof(Vertex));
	int mapIdx = 0;
	bool headingUp = true;
	float mapZoom = 1.6f;

	// state: start on the north-bound lane of the avenue at x = 0
	memset(cars, 0, sizeof cars);
	memset(&player, 0, sizeof player);
	carX = 3.0f; carZ = 30.0f; player.hp = 100.0f;
	float carY = 0.0f;
	float camX = 3.0f, camY = 3.5f, camZ = 38.0f;
	float sdirX = 0.0f, sdirZ = -1.0f;
	float cfx = 0.0f, cfz = -1.0f; // actual camera forward (for culling)
	float spawnT = 0.0f, bustT = 0.0f;
	u64 last = osGetTime();

	{
		StreamCtx s = { carX, carZ, camX, camZ, sdirX, sdirZ, KEEP_AHEAD };
		updateStreaming(&s, MAX_CHUNKS);
	}
	for (int i = 0; i < MAX_CIV * 8; i++) spawnCivilian(carX, carZ);

	while (aptMainLoop()) {
		hidScanInput();
		u32 kHeld = hidKeysHeld();
		u32 kDown = hidKeysDown();
		if (kDown & KEY_START) break;
		if (kDown & KEY_X) headingUp = !headingUp;
		if (kDown & KEY_SELECT) player.hp = 100.0f;   // repair

		u64 now = osGetTime();
		float dt = (now - last) / 1000.0f;
		last = now;
		if (dt > 0.1f) dt = 0.1f;
		gCopPhase = (int)((now / 180) & 1);

		// minimap zoom
		if (kHeld & KEY_R) mapZoom *= expf(1.3f * dt);
		if (kHeld & KEY_L) mapZoom *= expf(-1.3f * dt);
		mapZoom = clampf(mapZoom, 1.0f, 2.6f);

		// --- player car (arcade); damage lowers the top speed, 0 health = wrecked ---
		circlePosition cp;
		hidCircleRead(&cp);
		float steer = cp.dx / 156.0f;
		if (fabsf(steer) < 0.12f) steer = 0.0f;

		float maxSpeed = (player.hp > 0.0f) ? 45.0f * (0.45f + 0.55f * player.hp / 100.0f) : 5.0f;
		if (kHeld & KEY_A)      carSpeed += 22.0f * dt;
		else if (kHeld & KEY_B) carSpeed -= 30.0f * dt;
		else                    carSpeed -= carSpeed * 0.6f * dt;
		if (carSpeed > maxSpeed) carSpeed += (maxSpeed - carSpeed) * (1.0f - expf(-3.0f * dt));
		if (carSpeed < -14.0f) carSpeed = -14.0f;

		float grip = clampf(carSpeed / 10.0f, -1.0f, 1.0f);
		carHeading += steer * 1.8f * grip * dt;
		carHeading += player.spin * dt;                 // yaw from impacts
		player.spin *= expf(-4.0f * dt);

		float fx = sinf(carHeading), fz = -cosf(carHeading);
		carX += (fx * carSpeed + player.vx) * dt;       // forward motion + impact slide
		carZ += (fz * carSpeed + player.vz) * dt;
		player.vx *= expf(-3.0f * dt);
		player.vz *= expf(-3.0f * dt);

		// --- freeway barriers: stay on the deck when elevated ---
		if (carY > 1.2f) {
			int fi = (int)floorf(carX / CHUNK_SIZE + 0.5f);
			if (isFreewayX(fi)) {
				float lx = fi * CHUNK_SIZE, lim = FWY_HALF - 1.7f, ddx = carX - lx;
				if (ddx > lim)  { carX = lx + lim;  carSpeed *= expf(-3.0f * dt); }
				if (ddx < -lim) { carX = lx - lim;  carSpeed *= expf(-3.0f * dt); }
			}
		}

		// --- buildings, tunnel walls, river (hitting them at speed damages the car) ---
		pushOutOfSolids(&player, dt);
		if (player.hp < 0.0f) player.hp = 0.0f;

		// --- road surface height (ground or freeway deck), smoothed ---
		{
			float tY = surfaceY(carX, carZ, carY);
			carY += (tY - carY) * (1.0f - expf(-14.0f * dt));
		}

		// --- cars, police, collisions, heat ---
		if (hitCool > 0.0f) hitCool -= dt;
		updateCars(dt, carY);
		if (player.hp < 0.0f) player.hp = 0.0f;

		if (fabsf(carSpeed) > 36.0f && heat < 30.0f) heat += 4.0f * dt;     // speeding
		if (heat > 0.0f) {                                                 // cooling off
			if (gNearestCop > 100.0f)     heat -= 5.0f * dt;
			else if (gNearestCop > 60.0f) heat -= 1.0f * dt;
		}
		heat = clampf(heat, 0.0f, 100.0f);

		// busted: stopped next to a police car for a few seconds
		if (heat > 0.0f && gNearestCop < 8.0f && fabsf(carSpeed) < 5.0f) bustT += dt;
		else bustT = fmaxf(0.0f, bustT - dt);
		if (bustT > 2.5f) { bustedT = 3.0f; bustT = 0.0f; heat = 0.0f; }
		if (bustedT > 0.0f) bustedT -= dt;

		// spawns
		spawnT -= dt;
		int level = heat > 0.0f ? (int)ceilf(heat / 20.0f) : 0;
		if (level > MAX_COP) level = MAX_COP;
		if (gCopCount < level && spawnT <= 0.0f) { spawnCop(carX, carZ); spawnT = 1.5f; }
		if (gCivCount < MAX_CIV) spawnCivilian(carX, carZ);

		// --- chase camera (hold Y for rear view) ---
		float side = (kHeld & KEY_Y) ? -1.0f : 1.0f;
		float wantX = carX - fx * 8.0f * side;
		float wantZ = carZ - fz * 8.0f * side;
		float wantY = carY + 3.5f;
		float k = 1.0f - expf(-6.0f * dt);
		camX += (wantX - camX) * k;
		camY += (wantY - camY) * k;
		camZ += (wantZ - camZ) * k;

		// --- streaming direction = smoothed camera facing ---
		float fdx = carX - camX, fdz = carZ - camZ;
		float fl = hypotf(fdx, fdz);
		if (fl > 0.01f) {
			fdx /= fl; fdz /= fl;
			cfx = fdx; cfz = fdz;
			float k2 = 1.0f - expf(-1.5f * dt);
			sdirX += (fdx - sdirX) * k2;
			sdirZ += (fdz - sdirZ) * k2;
			float sl = hypotf(sdirX, sdirZ);
			if (sl > 0.001f) { sdirX /= sl; sdirZ /= sl; }
		}
		StreamCtx s = { carX, carZ, camX, camZ, sdirX, sdirZ, KEEP_AHEAD + fabsf(carSpeed) * 0.5f };
		updateStreaming(&s, 1);

		// --- car pitch follows ramps ---
		float pitch = atan2f(surfaceY(carX + fx * 3.0f, carZ + fz * 3.0f, carY) -
		                     surfaceY(carX - fx * 3.0f, carZ - fz * 3.0f, carY), 6.0f);
		pitch = clampf(pitch, -0.4f, 0.4f);

		// --- build this frame's minimap geometry ---
		mapIdx ^= 1;
		buildMinimap(mapBuf[mapIdx], carX, carZ, carHeading, carSpeed, headingUp, mapZoom);
		GSPGPU_FlushDataCache(mapBuf[mapIdx], mapN * sizeof(Vertex));

		// --- render ---
		Mtx_PerspTilt(&projection, C3D_AngleFromDegrees(60.0f + fabsf(carSpeed) * 0.3f),
		              C3D_AspectRatioTop, 0.5f, 400.0f, false);
		Mtx_LookAt(&view, FVec3_New(camX, camY, camZ),
		           FVec3_New(carX, carY + 1.2f, carZ), FVec3_New(0, 1, 0), false);

		C3D_FrameBegin(C3D_FRAME_SYNCDRAW);

		// top screen: 3D city
		C3D_RenderTargetClear(top, C3D_CLEAR_ALL, CLEAR_COLOR, 0);
		C3D_FrameDrawOn(top);
		C3D_DepthTest(true, GPU_GREATER, GPU_WRITE_ALL);
		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_projection, &projection);

		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_modelView, &view);
		for (int i = 0; i < MAX_CHUNKS; i++) {
			if (!chunks[i].used) continue;
			float ccx = (chunks[i].cx + 0.5f) * CHUNK_SIZE - camX;
			float ccz = (chunks[i].cz + 0.5f) * CHUNK_SIZE - camZ;
			if (ccx * cfx + ccz * cfz < -46.0f) continue;
			bindVerts(chunks[i].verts);
			C3D_DrawArrays(GPU_TRIANGLES, 0, chunks[i].nverts);
		}

		// player
		C3D_Mtx mv;
		Mtx_Copy(&mv, &view);
		Mtx_Translate(&mv, carX, carY, carZ, true);
		Mtx_RotateY(&mv, -carHeading, true);
		Mtx_RotateX(&mv, pitch, true);
		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_modelView, &mv);
		int pm = (player.hp <= 0.0f) ? 7 : 0;
		bindVerts(carMesh[pm]);
		C3D_DrawArrays(GPU_TRIANGLES, 0, carMeshN[pm]);

		// civilians and police
		for (int i = 0; i < MAX_CARS; i++) {
			if (!cars[i].active) continue;
			int mi;
			if (cars[i].hp <= 0.0f)           mi = 7;
			else if (cars[i].kind == K_COP)   mi = 8 + gCopPhase;
			else                              mi = cars[i].color;
			Mtx_Copy(&mv, &view);
			Mtx_Translate(&mv, cars[i].x, 0.0f, cars[i].z, true);
			Mtx_RotateY(&mv, -cars[i].heading, true);
			C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_modelView, &mv);
			bindVerts(carMesh[mi]);
			C3D_DrawArrays(GPU_TRIANGLES, 0, carMeshN[mi]);
		}

		// bottom screen: 2D minimap and HUD (painter's order, no depth test)
		C3D_RenderTargetClear(bot, C3D_CLEAR_ALL, MAP_CLEAR, 0);
		C3D_FrameDrawOn(bot);
		C3D_DepthTest(false, GPU_ALWAYS, GPU_WRITE_COLOR);
		Mtx_OrthoTilt(&mapProj, 0.0f, 320.0f, 240.0f, 0.0f, 1.0f, -1.0f, true);
		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_projection, &mapProj);
		Mtx_Identity(&mv);
		C3D_FVUnifMtx4x4(GPU_VERTEX_SHADER, uLoc_modelView, &mv);
		bindVerts(mapBuf[mapIdx]);
		C3D_DrawArrays(GPU_TRIANGLES, 0, mapN);

		C3D_FrameEnd(0);
	}

	for (int i = 0; i < MAX_CHUNKS; i++) if (chunks[i].used) unloadChunk(&chunks[i]);
	for (int i = 0; i < NUM_MESHES; i++) linearFree(carMesh[i]);
	linearFree(mapBuf[0]);
	linearFree(mapBuf[1]);
	shaderProgramFree(&program);
	DVLB_Free(dvlb);
	C3D_Fini();
	gfxExit();
	return 0;
}
