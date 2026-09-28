"""
productions.py — Real filming locations from three real productions.

Three productions were chosen so the same scheduler can be tested at three
different geographic scales:

    scale                     production              locations   groups
    -----------------------   ---------------------   ---------   ----------------
    one city, many districts  La La Land (2016)             26    LA neighbourhoods
    one country, many cities  Forrest Gump (1994)           30    US states/towns
    many countries            Tenet (2020)                  34    countries/cities

The `group` field on each location records the district / town / city it truly
belongs to.  The scheduler never sees it; it exists only so the grouping step
can be scored against ground truth.

Provenance and its limits
-------------------------
Location lists are transcribed from published location guides (see SOURCES),
which are enthusiast reconstructions rather than production paperwork.  Every
entry was then checked independently against OpenStreetMap, a terrain model and,
where one exists, an authoritative record (National Register listings, park
service pages, municipal boundaries).  That pass changed the data in three ways:

  * three entries were removed for not being filming locations at all — the
    Bradbury Building (the guide says Grand Central Market is *across from* it),
    a Big Bear Lake museum (it rented the projector; the planetarium was built
    on a soundstage) and the Savannah History Museum (it displays a 1995 replica
    of the bench, which was never shot there);
  * twenty-three coordinates or elevations were corrected, the largest by 18 km
    and one of which had placed a location in open tidal water;
  * nine group labels were corrected, including a location listed in Arizona
    that is in Utah, and two groups that spanned 100 km or more were split,
    since these labels are the ground truth the partitioning is scored against
    and a group that wide would flatter it.

After that pass coordinates are landmark-accurate for named landmarks and
street-accurate for entries given only as an address: median error 0.18 km, and
the handful above 500 m are area features — a wind farm, an island, a park
entrance — where a single coordinate is inherently approximate.  That is
immaterial next to the tens of kilometres between districts.

Real shooting order is not published for any of the three, so the schedules
here can be compared with each other but not against what the crews actually
did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from stripboard.geo.graph import SpatialGraph, TerrainType
from stripboard.geo.road_network import GeoNode, assemble_road_network

U, F, M, C, D = (TerrainType.URBAN, TerrainType.FOREST, TerrainType.MOUNTAIN,
                 TerrainType.COASTAL, TerrainType.DESERT)

SOURCES = {
    "la_la_land": "movie-locations.com/movies/l/La-La-Land.php",
    "forrest_gump": "movie-locations.com/movies/f/Forrest-Gump.php",
    "tenet": "movie-locations.com/movies/t/Tenet-film-locations.php",
}


@dataclass
class Production:
    """One film's shooting locations, ready to be turned into a graph."""
    title: str
    scale: str
    source: str
    # (name, group, terrain, elevation_m, lat, lon) — first entry is the base.
    locations: List[Tuple[str, str, TerrainType, float, float, float]] = \
        field(default_factory=list)


PRODUCTIONS: Dict[str, Production] = {

    # -----------------------------------------------------------------------
    # One city, many districts.  All inside Los Angeles County, but spread over
    # roughly 60 km of it: West Hills sits at the north-west edge and Long Beach
    # at the south-east, far enough apart that the order they are shot in is a
    # real decision rather than a rounding error.
    # -----------------------------------------------------------------------
    "la_la_land": Production(
        title="La La Land (2016)",
        scale="one city, many districts",
        source=SOURCES["la_la_land"],
        locations=[
            ("warner_bros_coffee_shop",  "Burbank",        U,  152, 34.1483, -118.3370),
            ("warner_bros_studio_tour",  "Burbank",        U,  152, 34.1478, -118.3378),
            ("smokehouse_restaurant",    "Burbank",        U,  152, 34.1497, -118.3400),
            ("van_beek_magnolia_blvd",   "Burbank",        U,  158, 34.1665, -118.3520),
            ("retro_dairy_mart",         "Burbank",        U,  158, 34.1663, -118.3525),

            ("liptons_wilcox_ave",       "Hollywood",      U,  105, 34.1010, -118.3330),
            ("you_are_the_star_mural",   "Hollywood",      U,  105, 34.1016, -118.3330),
            ("sebs_club_exterior",       "Hollywood",      U,   90, 34.0900, -118.3230),

            ("griffith_observatory",     "Griffith Park",  M,  351, 34.1184, -118.3004),
            ("cathys_corner",            "Griffith Park",  M,  368, 34.1393, -118.3082),
            ("fern_dell",                "Griffith Park",  F,  180, 34.1090, -118.3100),

            ("grand_central_market",     "Downtown LA",    U,   90, 34.0505, -118.2487),
            ("angels_flight_railway",    "Downtown LA",    U,   95, 34.0513, -118.2489),

            ("jar_restaurant",           "Midtown LA",     U,   70, 34.0760, -118.3690),
            ("el_rey_theatre",           "Midtown LA",     U,   60, 34.0624, -118.3510),
            ("fais_do_do",               "Midtown LA",     U,   50, 34.0327, -118.3571),

            ("hermosa_beach_pier",       "Hermosa Beach",  C,    2, 33.8622, -118.4029),
            ("lighthouse_cafe",          "Hermosa Beach",  C,    3, 33.8620, -118.4005),

            ("rose_towers_apartment",    "Long Beach",     U,    8, 33.7691, -118.1705),
            ("the_blind_donkey",         "Long Beach",     U,    8, 33.7690, -118.1890),

            ("rialto_theatre",           "South Pasadena",       U,  194, 34.1140, -118.1508),
            ("colorado_street_bridge",   "Pasadena",       U,  210, 34.1444, -118.1650),

            ("freeway_overpass_105_110", "South LA",       U,   30, 33.9280, -118.2820),
            ("watts_towers",             "South LA",       U,   30, 33.9387, -118.2414),

            ("chateau_marmont",          "West Hollywood", U,  100, 34.0977, -118.3717),
            ("orcutt_ranch",             "West Hills",    F,  250, 34.2200, -118.6260),
        ],
    ),

    # -----------------------------------------------------------------------
    # One country, many towns.  Two tight clusters (the Beaufort area and
    # Savannah) plus points scattered from Maine to Montana.
    # -----------------------------------------------------------------------
    "forrest_gump": Production(
        title="Forrest Gump (1994)",
        scale="one country, many cities",
        source=SOURCES["forrest_gump"],
        locations=[
            ("chippewa_square",          "Savannah GA",    U,   12, 32.0777, -81.0930),

            ("uscb_performing_arts",     "Beaufort SC",    U,    5, 32.4316, -80.6698),
            ("woods_memorial_bridge",    "Beaufort SC",    C,    3, 32.4340, -80.6760),
            ("ladys_island_lucy_point",  "Beaufort SC",    C,    2, 32.4872, -80.6036),
            ("chowan_creek_bridge",      "Beaufort SC",    C,    1, 32.3979, -80.6159),
            ("port_royal",               "Beaufort SC",    C,    3, 32.3790, -80.6930),
            ("hunting_island_state_park","Beaufort SC",    C,    3, 32.3750, -80.4400),
            ("fripp_island",             "Beaufort SC",    C,    2, 32.3260, -80.4790),

            ("hampton_street_school",    "Walterboro SC",  U,   25, 32.9050, -80.6670),
            ("bluff_plantation",         "Yemassee SC",    F,    7, 32.6865, -80.7403),
            ("stoney_creek_chapel",      "McPhersonville SC", F, 25, 32.6921, -80.9147),
            ("varnville_route_68",       "Varnville SC",   U,   30, 32.8510, -81.0790),

            ("weingart_stadium",         "Monterey Park CA", U,   83, 34.0384, -118.1472),
            ("usc_marks_hall",           "Los Angeles CA", U,   50, 34.0220, -118.2860),
            ("usc_bovard_building",      "Los Angeles CA", U,   50, 34.0208, -118.2850),
            ("ebell_of_los_angeles",     "Los Angeles CA", U,   60, 34.0610, -118.3210),
            ("coles_pacific_electric",   "Los Angeles CA", U,   90, 34.0440, -118.2500),
            ("santa_monica_yacht_harbor","Santa Monica CA", C,    2, 34.0080, -118.4980),

            ("lincoln_memorial",         "Washington DC",  U,   10, 38.8893,  -77.0502),
            ("reflecting_pool",          "Washington DC",  U,   10, 38.8893,  -77.0430),
            ("jefferson_memorial",       "Washington DC",  U,    5, 38.8814,  -77.0365),
            ("watergate_hotel",          "Washington DC",  U,   10, 38.9000,  -77.0555),

            ("grandfather_mountain",     "Linville NC",    M, 1472, 36.0921,  -81.8357),

            ("downtown_flagstaff",       "Flagstaff AZ",   U, 2106, 35.1980, -111.6510),
            ("twin_arrows_trading_post", "Flagstaff AZ",   D, 1795, 35.1611, -111.2794),
            ("monument_valley",          "Monument Valley UT", D, 1560, 37.1016, -109.9907),

            ("marshall_point_lighthouse","Port Clyde ME",  C,   10, 43.9180,  -69.2610),

            ("cut_bank",                 "Cut Bank MT",     U, 1151, 48.6330, -112.3260),
            ("glacier_st_mary_entrance", "Glacier NP MT",     M, 1370, 48.7470, -113.4370),
            ("going_to_the_sun_road",    "Glacier NP MT",     M, 1400, 48.6700, -113.5420),
        ],
    ),

    # -----------------------------------------------------------------------
    # Seven countries.  Tallinn alone holds eleven locations, so the shape is
    # one very heavy block plus long intercontinental hops.
    # -----------------------------------------------------------------------
    "tenet": Production(
        title="Tenet (2020)",
        scale="many countries",
        source=SOURCES["tenet"],
        locations=[
            ("linnahall",                "Tallinn EE",     U,    5, 59.4450,  24.7480),
            ("telliskivi_creative_city", "Tallinn EE",     U,   15, 59.4390,  24.7290),
            ("liivalaia_courthouse",     "Tallinn EE",     U,   15, 59.4300,  24.7580),
            ("kumu_art_museum",          "Tallinn EE",     U,   25, 59.4380,  24.7900),
            ("hilton_tallinn_park",      "Tallinn EE",     U,   20, 59.4310,  24.7700),
            ("parnu_maantee_argentiina", "Tallinn EE",     U,   20, 59.4291,  24.7446),
            ("laagna_tee",               "Tallinn EE",     U,   25, 59.4390,  24.8300),
            ("saarepiiga_bridge",        "Tallinn EE",     U,   25, 59.4402,  24.8441),
            ("port_paljassaare",         "Tallinn EE",     C,    3, 59.4602,  24.7072),
            ("suur_paala_ulemiste",      "Tallinn EE",     U,   35, 59.4297,  24.8147),
            ("maarjamae_memorial",       "Tallinn EE",     C,   25, 59.4570,  24.8100),

            ("nysted_wind_farm",         "Lolland DK",     C,    0, 54.5500,  11.7200),

            ("gateway_of_india",         "Mumbai IN",      C,    3, 18.9220,  72.8347),
            ("cafe_mondegar",            "Mumbai IN",      U,    8, 18.9220,  72.8320),
            ("royal_bombay_yacht_club",  "Mumbai IN",      U,    5, 18.9218,  72.8330),
            ("neelam_shree_vardhan",     "Mumbai IN",      U,   10, 18.9680,  72.8060),

            ("villa_cimbrone",           "Amalfi Coast IT", M,  350, 40.6444,  14.6111),
            ("hotel_caruso_ravello",     "Amalfi Coast IT", M,  365, 40.6500,  14.6120),
            ("amalfi_seafront",          "Amalfi Coast IT", C,    3, 40.6340,  14.6030),
            ("bella_baia_beach_maiori",  "Amalfi Coast IT", C,    2, 40.6417,  14.6625),

            ("oslo_opera_house",         "Oslo NO",        C,    3, 59.9075,  10.7530),
            ("tjuvholmen_alle",          "Oslo NO",        C,    3, 59.9080,  10.7210),
            ("the_thief_hotel",          "Oslo NO",        C,    3, 59.9075,  10.7215),

            ("reform_club_pall_mall",    "London UK",      U,   15, 51.5060,   -0.1360),
            ("national_liberal_club",    "London UK",      U,   10, 51.5060,   -0.1240),
            ("cannon_hall_hampstead",    "London UK",      U,  125, 51.5590,   -0.1780),
            ("locanda_locatelli",        "London UK",      U,   25, 51.5140,   -0.1590),
            ("berkeley_mews",            "London UK",      U,   25, 51.5154,   -0.1576),
            ("southampton_sailgp",       "Southampton UK", C,    5, 50.7700,   -1.2900),

            ("hawthorne_plaza_mall",     "Hawthorne US", U,   20, 33.9160, -118.3520),
            ("lax_terminal_connector",   "Los Angeles US", U,   38, 33.9420, -118.4080),
            ("warner_bros_stage_16",     "Los Angeles US", U,  152, 34.1483, -118.3370),
            ("victorville_logistics_airport", "Mojave US", D,  876, 34.5975, -117.3830),
            ("eagle_mountain",           "Joshua Tree US",      D,  416, 33.8500, -115.4800),
        ],
    ),
}


def build_production(key: str, inter_hub_links: int = 3
                     ) -> Tuple[SpatialGraph, List[GeoNode], Production]:
    """
    Turn one production's location list into a graph the schedulers can run on.

    Locations sharing a `group` are wired to each other directly; groups reach
    each other only through their first-listed location, so cross-country moves
    have to be routed.  Weights come from the geographic layer's formula.

    Args:
        key            : One of the keys in PRODUCTIONS.
        inter_hub_links: How many nearest groups each group connects to.

    Returns:
        (graph, nodes, production) — the road network as a SpatialGraph,
        the located nodes (with coordinates and true group), and the metadata.

    Raises:
        KeyError  : If `key` is not a known production.
        ValueError: If the resulting road network is disconnected.
    """
    prod = PRODUCTIONS[key]
    nodes = [
        GeoNode(id=i, name=name, terrain_type=terrain, elevation_m=elev,
                is_basecamp=(i == 0), lat=lat, lon=lon, city=group)
        for i, (name, group, terrain, elev, lat, lon) in enumerate(prod.locations)
    ]
    return assemble_road_network(nodes, inter_hub_links), nodes, prod


def subset_upto(nodes: List[GeoNode], limit: int) -> List[GeoNode]:
    """
    Take whole places until adding another would exceed `limit` locations.

    Whole places, not the first `limit` entries: half a city is not a smaller
    version of the same problem.
    """
    groups: Dict[str, List[GeoNode]] = {}
    for nd in nodes:
        groups.setdefault(nd.city, []).append(nd)
    picked: List[GeoNode] = []
    for members in groups.values():
        if len(picked) + len(members) <= limit:
            picked.extend(members)
    return [GeoNode(id=i, name=nd.name, terrain_type=nd.terrain_type,
                    elevation_m=nd.elevation_m, is_basecamp=(i == 0),
                    lat=nd.lat, lon=nd.lon, city=nd.city)
            for i, nd in enumerate(picked)]


def true_groups(nodes: List[GeoNode]) -> Dict[str, List[int]]:
    """Ground-truth grouping, for scoring what the MST step discovered."""
    groups: Dict[str, List[int]] = {}
    for nd in nodes:
        groups.setdefault(nd.city, []).append(nd.id)
    return groups


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stripboard.geo.dijkstra import shortest_path

    for key in PRODUCTIONS:
        graph, nodes, prod = build_production(key)
        groups = true_groups(nodes)
        edges = sum(1 for i in range(graph.n) for j in range(i + 1, graph.n)
                    if graph.weight(i, j) > 0)
        print("=" * 72)
        print(f"  {prod.title} — {prod.scale}")
        print("=" * 72)
        print(f"  {graph.n} locations, {len(groups)} groups, {edges} roads, "
              f"connected = {graph.is_connected()}")
        for name, members in groups.items():
            print(f"    {name:<22s} {len(members):>2d}  "
                  f"{', '.join(nodes[i].name for i in members[:3])}"
                  f"{' ...' if len(members) > 3 else ''}")
        far = max(range(graph.n),
                  key=lambda v: shortest_path(graph, 0, v)[1])
        path, cost = shortest_path(graph, 0, far)
        print(f"  furthest from base: {nodes[far].name} (cost {cost:,.0f}) via")
        print(f"    {' -> '.join(nodes[v].city for v in path)}")
        print()
