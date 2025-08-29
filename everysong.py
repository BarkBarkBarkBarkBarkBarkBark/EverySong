#!/usr/bin/env python3
"""
random_mb_to_spotify.py

Goal
-----
Create a Spotify playlist (~N tracks) that is as close as possible to a *true random,
cross-genre* sample by:
  1) Sampling random recordings from the public MusicBrainz Web Service (WS/2)
  2) Prefer-resolving each to Spotify by ISRC; fallback to artist+title fuzzy match
  3) Building a brand-new Spotify playlist and adding the resolved tracks

Environment (required)
----------------------
export SPOTIFY_CLIENT_ID="..."
export SPOTIFY_CLIENT_SECRET="..."
export SPOTIFY_REDIRECT_URI="http://localhost:8080/callback"   # any registered redirect
# Optional: set a fixed market for availability filtering (defaults to 'US')
export SPOTIFY_MARKET="US"

Install
-------
pip install spotipy requests tenacity

Usage
-----
python random_mb_to_spotify.py --n 1000 --name "True Random #001" --public false
"""

import os
import sys
import time
import math
import random
import argparse
import urllib.parse
from typing import Dict, Iterable, List, Optional, Tuple, Set

import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
load_dotenv()


# -------------------------------
# Config
# -------------------------------
MB_BASE = "https://musicbrainz.org/ws/2"
MB_RATE_LIMIT_DELAY = 1.1  # polite: 1 req/sec recommended; we use a bit more margin
MB_APP_NAME = "RandomMBToSpotify/1.0 (contact: you@example.com)"  # replace with your contact

# Lucene search in MusicBrainz caps usable offsets; we’ll stay within 0..25000 per query.
MB_MAX_OFFSET = 25000

# A broad spread of high-level tags to sample from (MusicBrainz tag facet).
# Using many tags lets us randomize across styles and eras without genre lock-in.
MB_TAGS = [
    "rock","pop","jazz","classical","hip hop","electronic","metal","punk","folk","country",
    "blues","soul","funk","reggae","latin","ambient","house","techno","trance","disco",
    "k-pop","j-pop","afrobeats","salsa","bachata","flamenco","bossa nova","samba","tango",
    "gospel","soundtrack","lo-fi","emo","grunge","shoegaze","industrial","idm","drum and bass",
    "dubstep","edm","garage","uk drill","afrobeat","highlife","cumbia","mariachi","reggaeton",
    "ska","new wave","synthpop","post-punk","post-rock","hardcore","trap","r&b","alt-rock",
    "indie","chillout","medieval","baroque","romantic","modern","minimalism","opera","chiptune"
]

# We’ll ask MB to include ISRCs + artist-credits for better matching.
MB_INC = "isrcs+artist-credits"


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
    # Ensure JSON format & inc fields
    qp = {"fmt": "json"}
    qp.update(params)
    # Send
    resp = requests.get(url, headers=mb_headers(), params=qp, timeout=30)
    # Rate-limit delay (politeness)
    time.sleep(MB_RATE_LIMIT_DELAY)
    resp.raise_for_status()
    return resp.json()


def pick_random_offset(max_offset: int = MB_MAX_OFFSET, page_size: int = 100) -> int:
    """Choose a random page offset (aligned to page_size) within MB’s safe search window."""
    # Choose a random page number then convert to offset
    max_page = max(0, max_offset // page_size)
    page = random.randint(0, max_page)
    return page * page_size


def mb_sample_recordings(batch_target: int) -> Iterable[dict]:
    """
    Yield random-ish MusicBrainz recordings, each with ISRCs and artist credit when available.

    Strategy:
      - Choose a random high-level tag (genre-ish)
      - Random offset in the first ~25k results (MB search practical window)
      - Query recordings that likely have audio (length between 30 sec and 9 min when known)
      - Request 100 at a time, then shuffle locally
    """
    page_size = 100
    while True:
        tag = random.choice(MB_TAGS)
        offset = pick_random_offset()
        # Lucene query: tag:"<tag>" AND recording:*
        # We also *prefer* items with length known and not too tiny/long, but length is not an indexed field in MB search.
        # We'll filter length after we fetch.
        q = f'tag:"{tag}" AND recording:*'
        params = {
            "query": q,
            "inc": MB_INC,
            "limit": str(page_size),
            "offset": str(offset),
        }
        data = mb_get("recording", params)
        recs = data.get("recordings", []) or []

        # Local shuffle to reduce any positional bias
        random.shuffle(recs)

        emitted = 0
        for r in recs:
            # Optional length filter if present (ms)
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
        # MB returns a list of ISRC strings; pick first
        isrc = r["isrcs"][0]

    title = r.get("title") or ""
    # Artist-credit may include join phrases; we’ll just collect plain names
    artist_names = []
    for ac in r.get("artist-credit", []):
        if isinstance(ac, dict) and "name" in ac:
            artist_names.append(ac["name"])
    length = r.get("length")
    length = int(length) if length is not None else None
    return isrc, title, artist_names, length


# -------------------------------
# Spotify
# -------------------------------
def spotify_client() -> spotipy.Spotify:
    scope = "playlist-modify-public playlist-modify-private"
    auth = SpotifyOAuth(scope=scope)
    return spotipy.Spotify(auth_manager=auth)


@retry(wait=wait_exponential(multiplier=1.0, min=1, max=30),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(RateLimit))
def sp_search(sp: spotipy.Spotify, q: str, market: Optional[str] = None, limit: int = 10) -> List[dict]:
    try:
        res = sp.search(q=q, type="track", market=market, limit=limit)
    except spotipy.exceptions.SpotifyException as e:
        # If we hit 429, raise RateLimit to trigger retry with backoff
        if getattr(e, "http_status", None) == 429:
            raise RateLimit("Spotify 429")
        raise
    return res.get("tracks", {}).get("items", [])


def resolve_spotify_track(sp: spotipy.Spotify,
                          isrc: Optional[str],
                          title: str,
                          artist_names: List[str],
                          length_ms: Optional[int],
                          market: Optional[str]) -> Optional[str]:
    """
    Return spotify:track URI or None.

    Order:
      1) Try exact ISRC match (fast & precise)
      2) Fallback to track:"..." artist:"..." text search, filter by duration ±3s
    """
    # 1) ISRC
    if isrc:
        items = sp_search(sp, q=f"isrc:{isrc}", market=market, limit=1)
        if items:
            return items[0]["uri"]

    # 2) Text fallback
    title_q = f'track:"{title}"' if title else ""
    artist_q = ""
    if artist_names:
        # Use the first credited artist for precision; if ambiguous, Spotify returns top matches
        artist_q = f' artist:"{artist_names[0]}"'
    q = (title_q + artist_q).strip()
    if not q:
        return None

    items = sp_search(sp, q=q, market=market, limit=10)
    if not items:
        return None

    if length_ms is None:
        return items[0]["uri"]

    # Filter by duration tolerance to reduce false matches
    for it in items:
        if abs(int(it["duration_ms"]) - int(length_ms)) <= 3000:
            return it["uri"]

    return items[0]["uri"]  # last resort


def create_playlist(sp: spotipy.Spotify, name: str, public: bool) -> Tuple[str, str]:
    me = sp.current_user()
    pl = sp.user_playlist_create(me["id"], name=name, public=public, description="True-random, cross-genre sample via MusicBrainz")
    return pl["id"], pl["external_urls"]["spotify"]


def add_tracks_batched(sp: spotipy.Spotify, playlist_id: str, uris: List[str]) -> None:
    # Spotify allows adding 100 items per call
    for i in range(0, len(uris), 100):
        chunk = uris[i:i+100]
        # spotipy handles retries for auth, but we still may see sporadic errors; keep it simple
        sp.playlist_add_items(playlist_id, chunk)
        time.sleep(0.1)


# -------------------------------
# Main workflow
# -------------------------------
def build_true_random_playlist(n: int, name: str, public: bool) -> str:
    sp = spotify_client()
    market = os.environ.get("SPOTIFY_MARKET", "US")

    target = n
    needed_candidates = math.ceil(n * 1.8)  # oversample to cover mismatches/unavailable
    seen_spotify_ids: Set[str] = set()
    uris: List[str] = []

    print(f"[MB] Sampling ~{needed_candidates} recordings to target {n} Spotify matches...")
    playlist_id, playlist_url = create_playlist(sp, name, public)
    print(f"[SPOTIFY] Created playlist: {playlist_url}")

    # Iterate random MB recordings and resolve to Spotify
    for r in mb_sample_recordings(batch_target=100):  # get ~100 per MB query
        isrc, title, artists, length_ms = recording_to_search_keys(r)

        uri = resolve_spotify_track(sp, isrc, title, artists, length_ms, market)
        if not uri:
            continue

        sp_id = uri.split(":")[-1]
        if sp_id in seen_spotify_ids:
            continue

        seen_spotify_ids.add(sp_id)
        uris.append(uri)

        # Add in small batches to make progress even if we stop early
        if len(uris) % 50 == 0:
            print(f"[SPOTIFY] Adding {len(uris)} tracks so far...")
            add_tracks_batched(sp, playlist_id, uris[-50:])

        if len(uris) >= target:
            break

        # If we haven’t collected enough, continue sampling
        if len(seen_spotify_ids) + 200 > needed_candidates and len(uris) < target:
            # Increase oversampling on the fly
            needed_candidates += 500

    # Final top-up add if the last chunk wasn’t a multiple of 50
    # (We might have already added the tail inside the loop; dedupe add is harmless but we’ll check.)
    current_count = 0
    try:
        # quick sanity: read back total (not strictly necessary)
        pl = sp.playlist_items(playlist_id, limit=1)
        total = pl.get("total", 0)
        current_count = int(total)
    except Exception:
        pass

    if len(uris) > current_count:
        print(f"[SPOTIFY] Finalizing: adding remaining {len(uris) - current_count} tracks...")
        add_tracks_batched(sp, playlist_id, uris[current_count:])

    print(f"[DONE] Playlist ready at: {playlist_url}  (tracks: ~{len(uris)})")
    return playlist_url


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create a Spotify playlist from random MusicBrainz recordings.")
    ap.add_argument("--n", type=int, default=1000, help="Number of tracks to add (default: 1000)")
    ap.add_argument("--name", type=str, default="True Random via MusicBrainz", help="Playlist name")
    ap.add_argument("--public", type=str, default="false", help="Make playlist public? (true/false)")
    return ap.parse_args()


def main():
    # Basic env validation
    missing = [k for k in ("SPOTIFY_CLIENT_ID","SPOTIFY_CLIENT_SECRET","SPOTIFY_REDIRECT_URI") if not os.environ.get(k)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    public = args.public.strip().lower() in ("1","true","yes","y")
    try:
        build_true_random_playlist(n=args.n, name=args.name, public=public)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
