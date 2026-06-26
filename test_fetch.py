from data.football_api import FootballAPIClient
from data.storage import BlobStore
class DummyBlob:
    def upload_json(self, c, k, d): pass
    def download_json(self, c, k): return None
c = FootballAPIClient("https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard", "", "", DummyBlob(), "")
data = c.fetch_matches()
import json
with open("test_out.json", "w") as f:
    json.dump(data, f, indent=2)
