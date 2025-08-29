#!/usr/bin/env python3
"""
random_mb_to_spotify.py

Goal
-----
Create a Spotify playlist (~N tracks) that is as close as possible to a *true random,
cross-genre, global* sample by:
  1) Sampling random recordings from the public MusicBrainz Web Service (WS/2)
  2) Prefer-resolving each to Spotify by ISRC; fallback to artist+title fuzzy match
  3) Building a brand-new Spotify playlist and adding the resolved tracks
  4) Rotating **Spotify markets** so results come from many countries, not just one

Environment (required)
----------------------
# NOTE (2025 redirect rules): use explicit loopback IP, not 'localhost'
export SPOTIPY_CLIENT_ID="..."        # or SPOTIFY_CLIENT_ID
export SPOTIPY_CLIENT_SECRET="..."    # or SPOTIFY_CLIENT_SECRET
export SPOTIPY_REDIRECT_URI="http://127.0.0.1:8080/callback"  # or http://[::1]:PORT
# Optional: set a default market if not using --markets
export SPOTIFY_MARKET="US"

Install
-------
pip install spotipy requests tenacity python-dotenv

Usage
-----
# US-only (or your env default)
python random_mb_to_spotify.py --n 1000 --name "Random US"

# True international rotation (recommended)
python random_mb_to_spotify.py --n 1000 --name "World Mix" --markets GLOBAL

# Custom market list
python random_mb_to_spotify.py --n 1000 --name "Americas" --markets "US,CA,MX,BR,AR,CL,CO"

# Control genre spread and artist clustering
python random_mb_to_spotify.py --n 1000 --name "Global Cross-Genre" \
  --markets GLOBAL --artist-cap 2 \
  --genres "pop,hip hop,edm,afrobeats,latin,k-pop,rock,jazz,classical"

# Limit by release year
python random_mb_to_spotify.py --n 1000 --name "Fresh 5 Years" --markets GLOBAL --year-min 2020
"""

import argparse
import itertools
import math
import os
import random
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Optional: load a local .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# -------------------------------
# Config
# -------------------------------
MB_BASE = "https://musicbrainz.org/ws/2"
MB_RATE_LIMIT_DELAY = 1.1  # polite: ~1 req/sec
MB_APP_NAME = "RandomMBToSpotify/1.3 (contact: you@example.com)"  # replace contact

# Lucene search in MusicBrainz caps usable offsets; stay within 0..25000 per query.
MB_MAX_OFFSET = 25000

# Expanded, modern-leaning tag set. You can override via --genres.
DEFAULT_GENRE_TAGS = [
    "pop",
    "hip hop",
    "rap",
    "trap",
    "r&b",
    "soul",
    "afrobeats",
    "afrobeat",
    "amapiano",
    "latin",
    "reggaeton",
    "regional mexican",
    "corrido",
    "banda",
    "salsa",
    "bachata",
    "k-pop",
    "j-pop",
    "city pop",
    "cantopop",
    "indie pop",
    "indie rock",
    "alt-rock",
    "edm",
    "electronic",
    "house",
    "deep house",
    "progressive house",
    "tech house",
    "techno",
    "hard techno",
    "trance",
    "psytrance",
    "drum and bass",
    "dubstep",
    "garage",
    "grime",
    "uk drill",
    "breakbeat",
    "lo-fi",
    "chillout",
    "ambient",
    "synthwave",
    "hyperpop",
    "rock",
    "classic rock",
    "punk",
    "post-punk",
    "shoegaze",
    "emo",
    "post-rock",
    "metal",
    "metalcore",
    "country",
    "folk",
    "bluegrass",
    "blues",
    "jazz",
    "fusion",
    "bossa nova",
    "samba",
    "mpb",
    "reggae",
    "dancehall",
    "ska",
    "gospel",
    "soundtrack",
    "classical",
    "contemporary classical",
    "opera",
]

# A broad, representative set of Spotify markets to rotate through for a global feel.
# (Not exhaustive, but wide coverage across regions.)
DEFAULT_MARKETS_GLOBAL = [
    "US",
    "CA",
    "MX",
    "BR",
    "AR",
    "CL",
    "CO",
    "PE",
    "VE",
    "GB",
    "IE",
    "DE",
    "FR",
    "NL",
    "BE",
    "LU",
    "CH",
    "AT",
    "IT",
    "ES",
    "PT",
    "SE",
    "NO",
    "DK",
    "FI",
    "PL",
    "CZ",
    "RO",
    "HU",
    "GR",
    "TR",
    "UA",
    "RU",  # may be restricted for some accounts; harmless if skipped by API
    "AU",
    "NZ",
    "JP",
    "KR",
    "HK",
    "TW",
    "SG",
    "MY",
    "TH",
    "VN",
    "PH",
    "ID",
    "IN",
    "AE",
    "SA",
    "EG",
    "ZA",
    "NG",
    "KE",
    "MA",
    "IL",
]
# Note: CN is not a valid Spotify market and is intentionally omitted.

# Ask MB to include ISRCs + artist-credits + releases (for dates).
MB_INC = "isrcs+artist-credits+releases"

# -------------------------------
# Helpers
# -------------------------------
class RateLimit(Exception):
    """Simple wrapper to trigger retry on Spotify 429."""

    pass


def mb_headers() -> Dict[str, str]:
    return {
        "User-Agent": MB_APP_NAME,
        "Accept": "application/json",
    }


def mb_get(path: str, params: Dict[str, str]) -> dict:
    """Polite GET to MusicBrainz with rate-limit delay and JSON parsing."""
    url = f"{MB_BASE}/{path}"
    qp = {"fmt": "json"}
    qp.update(params)
    resp = requests.get(url, headers=mb_headers(), params=qp, timeout=30)
    time.sleep(MB_RATE_LIMIT_DELAY)  # politeness
    resp.raise_for_status()
    return resp.json()


def pick_random_offset(max_offset: int = MB_MAX_OFFSET, page_size: int = 100) -> int:
    """Choose a random page offset (aligned to page_size) within MB’s safe search window."""
    max_page = max(0, max_offset // page_size)
    page = random.randint(0, max_page)
    return page * page_size


def normalize_tag_name(t: str) -> str:
    return t.strip().lower()


def recording_release_year(r: dict) -> Optional[int]:
    """Extract earliest release year from a recording JSON (if available)."""
    releases = r.get("releases") or []
    years: List[int] = []
    for rel in releases:
        date = rel.get("date")
        if not date:
            continue
        try:
            years.append(int(date.split("-")[0]))
        except (ValueError, AttributeError):
            continue
    return min(years) if years else None


# ---------- Genre-aware round-robin sampler ----------
def mb_fetch_by_tag(tag: str, page_size: int = 100) -> List[dict]:
    """Fetch one random page of recordings for a given tag."""
    tag = normalize_tag_name(tag)
    offset = pick_random_offset()
    q = f'tag:"{tag}" AND recording:*'
    params = {"query": q, "inc": MB_INC, "limit": str(page_size), "offset": str(offset)}
    data = mb_get("recording", params)
    recs = data.get("recordings", []) or []
    random.shuffle(recs)
    return recs


def mb_round_robin_by_tags(
    tags: List[str],
    batch_per_tag: int = 20,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Iterable[dict]:
    """
    Yield recordings by cycling tags; from each tag page, emit up to batch_per_tag items
    that pass lightweight filters (length + optional year range), then move to next tag.
    """
    tags = [normalize_tag_name(t) for t in tags if t.strip()]
    if not tags:
        tags = DEFAULT_GENRE_TAGS[:]

    # Randomize order each full cycle to reduce residual bias
    for tag in itertools.cycle(random.sample(tags, k=len(tags))):
        try:
            recs = mb_fetch_by_tag(tag)
        except Exception:
            continue  # transient MB error; skip this turn

        emitted = 0
        for r in recs:
            # length filter (ms)
            length = r.get("length")
            if length is not None and not (30_000 <= int(length) <= 540_000):
                continue

            # year filter (inclusive). If filtering requested and year unknown, skip.
            if year_min is not None or year_max is not None:
                y = recording_release_year(r)
                if y is None:
                    continue
                if year_min is not None and y < year_min:
                    continue
                if year_max is not None and y > year_max:
                    continue

            yield r
            emitted += 1
            if emitted >= batch_per_tag:
                break


# Keep the old "free random" sampler around (not used by default anymore)
def mb_sample_recordings(batch_target: int) -> Iterable[dict]:
    """Legacy sampler: random tag + random offset within a tag slice."""
    page_size = 100
    tags = DEFAULT_GENRE_TAGS
    while True:
        tag = random.choice(tags)
        offset = pick_random_offset()
        q = f'tag:"{tag}" AND recording:*'
        params = {"query": q, "inc": MB_INC, "limit": str(page_size), "offset": str(offset)}
        data = mb_get("recording", params)
        recs = data.get("recordings", []) or []
        random.shuffle(recs)
        emitted = 0
        for r in recs:
            length = r.get("length")
            if length is not None and not (30_000 <= int(length) <= 540_000):
                continue
            yield r
            emitted += 1
            if emitted >= batch_target:
                break


def recording_to_search_keys(r: dict) -> Tuple[Optional[str], str, List[str], Optional[int]]:
    """Extract ISRC (first if many), title, artist names, and length (ms) from MB recording JSON."""
    isrc = None
    if "isrcs" in r and r["isrcs"]:
        isrc = r["isrcs"][0]
    title = r.get("title") or ""
    artist_names: List[str] = []
    for ac in r.get("artist-credit", []):
        if isinstance(ac, dict) and "name" in ac:
            artist_names.append(ac["name"])
    length = r.get("length")
    length_ms = int(length) if length is not None else None
    return isrc, title, artist_names, length_ms


# -------------------------------
# Spotify
# -------------------------------
def getenv_either(*names: str, required: bool = False) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    if required:
        raise RuntimeError(f"Missing required env var: one of {', '.join(names)}")
    return None


def spotify_client() -> spotipy.Spotify:
    # Accept either SPOTIPY_* (what Spotipy itself expects) or SPOTIFY_*
    client_id = getenv_either("SPOTIPY_CLIENT_ID", "SPOTIFY_CLIENT_ID", required=True)
    client_secret = getenv_either("SPOTIPY_CLIENT_SECRET", "SPOTIFY_CLIENT_SECRET", required=True)
    redirect_uri = getenv_either("SPOTIPY_REDIRECT_URI", "SPOTIFY_REDIRECT_URI", required=True)
    scope = "playlist-modify-public playlist-modify-private"
    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )
    return spotipy.Spotify(auth_manager=auth)


@retry(
    wait=wait_exponential(multiplier=1.0, min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimit),
)
def sp_search(
    sp: spotipy.Spotify, q: str, market: Optional[str] = None, limit: int = 10
) -> List[dict]:
    try:
        res = sp.search(q=q, type="track", market=market, limit=limit)
    except spotipy.exceptions.SpotifyException as e:
        if getattr(e, "http_status", None) == 429:
            raise RateLimit("Spotify 429")
        raise
    return res.get("tracks", {}).get("items", [])


def resolve_spotify_track_global(
    sp: spotipy.Spotify,
    isrc: Optional[str],
    title: str,
    artist_names: List[str],
    length_ms: Optional[int],
    markets: List[str],
) -> Optional[str]:
    """
    Try to resolve a track across a list of Spotify markets.
    We randomize market order each call to spread coverage.
    """
    markets = [m for m in markets if m]  # filter blanks
    if not markets:
        markets = ["US"]

    shuffled = markets[:]
    random.shuffle(shuffled)

    # 1) ISRC across markets
    if isrc:
        for m in shuffled:
            items = sp_search(sp, q=f"isrc:{isrc}", market=m, limit=1)
            if items:
                return items[0]["uri"]

    # 2) Text fallback across markets
    title_q = f'track:"{title}"' if title else ""
    artist_q = f' artist:"{artist_names[0]}"' if artist_names else ""
    q = (title_q + artist_q).strip()
    if not q:
        return None

    for m in shuffled:
        items = sp_search(sp, q=q, market=m, limit=10)
        if not items:
            continue
        if length_ms is None:
            return items[0]["uri"]
        for it in items:
            if abs(int(it["duration_ms"]) - int(length_ms)) <= 3000:
                return it["uri"]
        # otherwise return first from this market as last resort
        return items[0]["uri"]

    return None


def create_playlist(sp: spotipy.Spotify, name: str, public: bool) -> Tuple[str, str]:
    me = sp.current_user()
    pl = sp.user_playlist_create(
        me["id"],
        name=name,
        public=public,
        description="True-random, genre-rotating, multi-market sample via MusicBrainz",
    )
    return pl["id"], pl["external_urls"]["spotify"]


def add_tracks_batched(sp: spotipy.Spotify, playlist_id: str, uris: List[str]) -> None:
    # Spotify allows adding 100 items per call
    for i in range(0, len(uris), 100):
        chunk = uris[i : i + 100]
        sp.playlist_add_items(playlist_id, chunk)
        time.sleep(0.1)


# -------------------------------
# Main workflow
# -------------------------------
def parse_markets_arg(markets_csv: Optional[str]) -> List[str]:
    """
    Interpret --markets:
      - None/""  -> use env SPOTIFY_MARKET or default ["US"]
      - "GLOBAL" -> DEFAULT_MARKETS_GLOBAL
      - otherwise comma-separated list of ISO country codes
    """
    if markets_csv and markets_csv.strip():
        if markets_csv.strip().upper() == "GLOBAL":
            return DEFAULT_MARKETS_GLOBAL[:]
        return [m.strip().upper() for m in markets_csv.split(",") if m.strip()]
    # Fallbacks
    env_m = os.getenv("SPOTIFY_MARKET")
    return [env_m.upper()] if env_m else ["US"]


def build_true_random_playlist(
    n: int,
    name: str,
    public: bool,
    genres_csv: str,
    per_genre_batch: int,
    artist_cap: int,
    markets_csv: Optional[str],
    year_min: Optional[int],
    year_max: Optional[int],
) -> str:
    sp = spotify_client()
    markets = parse_markets_arg(markets_csv)

    playlist_id, playlist_url = create_playlist(sp, name, public)
    print(f"[SPOTIFY] Created playlist: {playlist_url}")
    preview = ", ".join(markets[:12]) + (" ..." if len(markets) > 12 else "")
    print(f"[INFO] Resolving across markets: {preview}")

    target = n
    seen_spotify_ids: Set[str] = set()
    per_artist_count: Dict[str, int] = {}
    uris: List[str] = []

    # Parse & sanitize genre tags
    chosen_tags = [t.strip() for t in (genres_csv or "").split(",") if t.strip()]
    if not chosen_tags:
        chosen_tags = DEFAULT_GENRE_TAGS[:]

    # Round-robin generator across tags (with year filters)
    rr_iter = mb_round_robin_by_tags(
        chosen_tags,
        batch_per_tag=per_genre_batch,
        year_min=year_min,
        year_max=year_max,
    )

    for r in rr_iter:
        isrc, title, artists, length_ms = recording_to_search_keys(r)
        if not title:
            continue

        # Basic primary-artist heuristic for capping
        primary_artist = (artists[0].lower() if artists else "").strip()

        # Enforce artist cap
        if primary_artist and per_artist_count.get(primary_artist, 0) >= artist_cap:
            continue

        uri = resolve_spotify_track_global(sp, isrc, title, artists, length_ms, markets)
        if not uri:
            continue

        sp_id = uri.split(":")[-1]
        if sp_id in seen_spotify_ids:
            continue

        # Count per-artist and accept
        if primary_artist:
            per_artist_count[primary_artist] = per_artist_count.get(primary_artist, 0) + 1

        seen_spotify_ids.add(sp_id)
        uris.append(uri)

        # Add in small batches so progress is saved
        if len(uris) % 50 == 0:
            print(f"[SPOTIFY] Adding {len(uris)} tracks so far...")
            add_tracks_batched(sp, playlist_id, uris[-50:])

        if len(uris) >= target:
            break

    # Final flush if needed
    remaining = len(uris) % 50
    if remaining:
        print(f"[SPOTIFY] Finalizing: adding remaining {remaining} tracks...")
        add_tracks_batched(sp, playlist_id, uris[-remaining:])

    print(f"[DONE] Playlist ready at: {playlist_url}  (tracks: ~{len(uris)})")
    return playlist_url


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a Spotify playlist from random MusicBrainz recordings."
    )
    ap.add_argument("--n", type=int, default=1000, help="Number of tracks to add.")
    ap.add_argument("--name", type=str, default="True Random via MusicBrainz", help="Playlist name.")
    ap.add_argument("--public", type=str, default="false", help="Make playlist public? (true/false)")
    # Genre balancing
    ap.add_argument(
        "--genres",
        type=str,
        default=",".join(DEFAULT_GENRE_TAGS),
        help="Comma-separated MusicBrainz tags to sample uniformly across.",
    )
    ap.add_argument(
        "--per-genre-batch",
        type=int,
        default=20,
        help="Candidates to emit per tag each round (default: 20).",
    )
    ap.add_argument("--artist-cap", type=int, default=2, help="Max tracks per primary artist (default: 2).")
    # Markets
    ap.add_argument(
        "--markets",
        type=str,
        default=None,
        help=(
            'Comma-separated list of ISO markets to rotate (e.g., "US,GB,BR,JP") '
            'or "GLOBAL" for a wide preset. If omitted, uses $SPOTIFY_MARKET or US.'
        ),
    )
    # Year range
    ap.add_argument("--year-min", type=int, default=None, help="Earliest release year (inclusive).")
    ap.add_argument("--year-max", type=int, default=None, help="Latest release year (inclusive).")
    return ap.parse_args()


def main() -> None:
    # Validate env (accept SPOTIPY_* or SPOTIFY_*)
    for group in (
        ("SPOTIPY_CLIENT_ID", "SPOTIFY_CLIENT_ID"),
        ("SPOTIPY_CLIENT_SECRET", "SPOTIFY_CLIENT_SECRET"),
        ("SPOTIPY_REDIRECT_URI", "SPOTIFY_REDIRECT_URI"),
    ):
        if not getenv_either(*group):
            print(f"Missing required env var: one of {group[0]} or {group[1]}", file=sys.stderr)
            sys.exit(1)

    args = parse_args()
    public = args.public.strip().lower() in ("1", "true", "yes", "y")

    try:
        build_true_random_playlist(
            n=args.n,
            name=args.name,
            public=public,
            genres_csv=args.genres,
            per_genre_batch=args.per_genre_batch,
            artist_cap=args.artist_cap,
            markets_csv=args.markets,
            year_min=args.year_min,
            year_max=args.year_max,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
