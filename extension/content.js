async function fetchLiveMatch() {
  try {
    const res = await fetch("https://site.web.api.espn.com/apis/v2/sports/cricket/ipl/scoreboard");
    const data = await res.json();

    const events = data.events;
    if (!events || events.length === 0) return null;

    let liveMatch = events.find(e => e.status.type.state === "in");
    if (!liveMatch) return null;

    const comp = liveMatch.competitions[0];
    const teams = comp.competitors;

    const battingTeam = teams[0];

    return {
      live: true,
      striker: "Unknown",
      bowler: "Unknown",
      score: parseInt(battingTeam.score) || 0,
      wickets: 0,
      over: parseInt(comp.status.period) || 0,
      balls: 0,
      venue: comp.venue.fullName
    };

  } catch (e) {
    console.log(e);
    return null;
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "GET_DATA") {
    fetchLiveMatch().then(data => {
      sendResponse(data || { live: false });
    });
    return true;
  }
});