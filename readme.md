# 🎲 Random MusicBrainz → Spotify Playlist Generator

Ever wanted a **completely random, cross-genre** Spotify playlist — not biased to one style, not just radio, but pulled from across the world’s music catalog?

This tool builds playlists by:
1. Sampling random **recordings from [MusicBrainz](https://musicbrainz.org/)** (community-curated global music DB).
2. Resolving each track to **Spotify** via ISRC or fuzzy artist+title matching.
3. Uploading them into a brand-new Spotify playlist (your account).

Think of it as your personal **“shuffle all of music”** button. ✨

---

## 🚀 Features
- 🎧 Generates playlists up to **10,000 songs** (Spotify limit).
- 🌍 **Cross-genre randomness** via MusicBrainz tag sampling (pop, hip hop, EDM, afrobeats, jazz, classical, metal…).
- 🔄 **Round-robin genre cycling** → avoids being stuck in blues/oldies or any single tag.
- 👥 **Artist cap** option → no more 50 tracks from the same band.
- 🧩 Uses **ISRC codes** when possible for accurate matches, falls back gracefully.
- 📦 Simple Python script — no servers, no hacks.

---

## ⚙️ Requirements
- Python 3.9+  
- A [Spotify Developer App](https://developer.spotify.com/dashboard/)  
- Dependencies:
  ```bash
  pip install spotipy requests tenacity python-dotenv
# EverySong

```
python everysong.py --n 100 --name "Every Song Dot Com"  
```