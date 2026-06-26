/**
 * KickCast Almanac — Frontend JS
 * Fetches JSON predictions from the backend and builds the DOM.
 */

const COUNTRY_CODES = {
    "Argentina": "ar", "Australia": "au", "Belgium": "be", "Brazil": "br",
    "Cameroon": "cm", "Canada": "ca", "Colombia": "co", "Costa Rica": "cr",
    "Croatia": "hr", "Denmark": "dk", "Ecuador": "ec", "England": "gb-eng",
    "France": "fr", "Germany": "de", "Ghana": "gh", "Iran": "ir",
    "Japan": "jp", "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "Poland": "pl", "Portugal": "pt", "Qatar": "qa", "Saudi Arabia": "sa",
    "Senegal": "sn", "Serbia": "rs", "South Korea": "kr", "Spain": "es",
    "Switzerland": "ch", "Tunisia": "tn", "United States": "us", "Uruguay": "uy",
    "Wales": "gb-wls", "Curaçao": "cw", "Ivory Coast": "ci", "Sweden": "se",
    "Paraguay": "py", "Türkiye": "tr", "Panama": "pa", "Scotland": "gb-sct",
    "South Africa": "za", "Czechia": "cz", "Haiti": "ht", "Austria": "at",
    "Italy": "it", "Slovakia": "sk", "Ukraine": "ua", "Peru": "pe",
    "Chile": "cl", "Venezuela": "ve", "Jamaica": "jm", "Nigeria": "ng",
    "Mali": "ml", "New Zealand": "nz", "Georgia": "ge", "Albania": "al",
    "Slovenia": "si", "Romania": "ro"
};

let allMatches = [];
let uniqueDates = [];
let currentDateStr = '';

document.addEventListener('DOMContentLoaded', () => {
    
    // Mobile menu toggle
    const menuToggle = document.getElementById('mobile-menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }

    // Current date might not exist in index.html if removed, checking just in case
    const dateEl = document.getElementById('current-date');
    if (dateEl) {
        const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        dateEl.textContent = new Date().toLocaleDateString('en-US', dateOptions);
    }

    document.getElementById('btn-prev').addEventListener('click', () => {
        const idx = uniqueDates.indexOf(currentDateStr);
        if (idx > 0) {
            currentDateStr = uniqueDates[idx - 1];
            renderSidebar();
            renderMatches();
        }
    });

    document.getElementById('btn-next').addEventListener('click', () => {
        const idx = uniqueDates.indexOf(currentDateStr);
        if (idx < uniqueDates.length - 1) {
            currentDateStr = uniqueDates[idx + 1];
            renderSidebar();
            renderMatches();
        }
    });

    fetchPredictions();
});

function getFlagUrl(teamName) {
    const code = COUNTRY_CODES[teamName];
    if (code) return `https://flagcdn.com/24x18/${code}.png`;
    return ""; // Fallback if no flag
}

async function fetchPredictions() {
    const tbody = document.getElementById('predictions-body');

    try {
        const response = await fetch('/api/matches');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        allMatches = data.matches || [];

        if (allMatches.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5">
                        <div class="empty-state">No upcoming fixtures scheduled at this time.</div>
                    </td>
                </tr>
            `;
            return;
        }

        renderSidebar();

        // Compute unique local dates
        const dateSet = new Set();
        allMatches.forEach(m => {
            const d = new Date(m.date);
            dateSet.add(d.toLocaleDateString('en-US'));
        });
        uniqueDates = Array.from(dateSet).sort((a, b) => new Date(a) - new Date(b));

        // Auto-navigate to today's or the next upcoming match
        const now = new Date();
        now.setHours(0, 0, 0, 0);
        
        let targetDateStr = uniqueDates.find(dStr => new Date(dStr) >= now);
        
        // If all matches are in the past, default to the last match
        if (!targetDateStr && uniqueDates.length > 0) {
            targetDateStr = uniqueDates[uniqueDates.length - 1];
        }
        
        currentDateStr = targetDateStr;

        renderSidebar();
        renderMatches();

    } catch (error) {
        console.error("Failed to load predictions:", error);
        tbody.innerHTML = `
            <tr>
                <td colspan="5">
                    <div class="empty-state" style="color: var(--accent-color);">
                        Unable to connect to the prediction engine.<br>
                        <small>${error.message}</small>
                    </div>
                </td>
            </tr>
        `;
    }
}

function renderSidebar() {
    const nav = document.getElementById('stage-nav');
    nav.innerHTML = '';
    
    // Hardcoded order to ensure correct flow
    const STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "3rd Place Match", "Final", "World Cup"];
    
    // Find unique stages present in data
    const stagesInData = new Set(allMatches.map(m => m.stage));
    
    const stagesToRender = STAGE_ORDER.filter(s => stagesInData.has(s));
    stagesInData.forEach(s => {
        if (!stagesToRender.includes(s)) stagesToRender.push(s);
    });

    // Find the stage for the currentDateStr
    let currentStage = 'Group Stage';
    const matchForDate = allMatches.find(m => new Date(m.date).toLocaleDateString('en-US') === currentDateStr);
    if (matchForDate) {
        currentStage = matchForDate.stage;
    }
    
    stagesToRender.forEach(stage => {
        const btn = document.createElement('button');
        btn.className = 'nav-link';
        btn.textContent = stage;
        if (stage === currentStage) {
            btn.classList.add('active');
        }
        
        btn.addEventListener('click', () => {
            // Find the first date for this stage
            const stageMatch = allMatches.find(m => m.stage === stage);
            if (stageMatch) {
                currentDateStr = new Date(stageMatch.date).toLocaleDateString('en-US');
            }
            
            document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            renderMatches();

            // Auto close mobile menu
            const sidebar = document.querySelector('.sidebar');
            if (sidebar && sidebar.classList.contains('active')) {
                sidebar.classList.remove('active');
            }
        });
        
        nav.appendChild(btn);
    });
}

function renderMatches() {
    document.body.classList.remove('mobile-expanded-mode');
    const tbody = document.getElementById('predictions-body');
    tbody.innerHTML = '';
    
    let filtered = allMatches.filter(m => new Date(m.date).toLocaleDateString('en-US') === currentDateStr);
    const paginationControls = document.getElementById('pagination-controls');
    
    paginationControls.classList.remove('hidden');
    
    const currentIndex = uniqueDates.indexOf(currentDateStr);
    const d = new Date(currentDateStr);
    document.getElementById('page-indicator').textContent = d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
    
    document.getElementById('btn-prev').disabled = currentIndex <= 0;
    document.getElementById('btn-next').disabled = currentIndex >= uniqueDates.length - 1;

    if (filtered.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5">
                    <div class="empty-state">No matches found for ${currentDateStr}.</div>
                </td>
            </tr>
        `;
        return;
    }

    filtered.forEach(match => {
        const tr = document.createElement('tr');
        tr.className = "match-row";
        
        const dateObj = new Date(match.date);
        const dateStr = dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + 
                        ', ' + 
                        dateObj.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

        const isPlaceholder = (team) => {
            if (!team) return true;
            const lower = team.toLowerCase();
            return lower.includes('winner') || lower.includes('loser') || lower.includes('runner-up') || lower.includes('group') || lower.includes('match') || lower.includes('1st') || lower.includes('2nd') || lower.includes('round') || lower.includes('semifinal') || lower.includes('quarterfinal') || lower.includes('final') || lower === 'tbd';
        };

        const hasPlaceholders = isPlaceholder(match.home_team) || isPlaceholder(match.away_team);

        let pHome = (match.prob_home * 100).toFixed(1);
        let pDraw = (match.prob_draw * 100).toFixed(1);
        let pAway = (match.prob_away * 100).toFixed(1);

        const getAbbr = (teamName) => {
            if (teamName === 'South Africa') return 'RSA';
            if (teamName === 'South Korea') return 'KOR';
            if (teamName === 'Costa Rica') return 'CRC';
            if (teamName === 'New Zealand') return 'NZL';
            if (teamName === 'United States') return 'USA';
            if (teamName === 'Saudi Arabia') return 'KSA';
            return teamName.substring(0,3).toUpperCase();
        };

        const homeFlag = getFlagUrl(match.home_team);
        const awayFlag = getFlagUrl(match.away_team);

        let probHtml = '';
        let pickHtml = '';
        if (hasPlaceholders) {
            probHtml = `<div class="empty-state" style="padding: 0; border: none; font-size: 0.85rem;">Awaiting Finalists...</div>`;
            pickHtml = `<span style="color: var(--text-muted); font-style: italic; font-size: 0.9rem;">TBD</span>`;
        } else {
            let extraProbText = '';
            if (match.stage !== 'Group Stage') {
                let aetTotal = ((match.aet_home_prob + match.aet_away_prob) * 100).toFixed(1);
                let penTotal = ((match.pen_home_prob + match.pen_away_prob) * 100).toFixed(1);
                if (aetTotal > 0 || penTotal > 0) {
                     extraProbText = `<div style="text-align:center; font-size:0.75rem; color:var(--text-muted); margin-top:0.4rem; font-weight: 500;">AET: ${aetTotal}% | Pens: ${penTotal}%</div>`;
                }
            }

            probHtml = `
                <div class="prob-bar-wrapper">
                    <div class="prob-segment home" style="width: ${pHome}%"></div>
                    <div class="prob-segment draw" style="width: ${pDraw}%"></div>
                    <div class="prob-segment away" style="width: ${pAway}%"></div>
                </div>
                <div class="prob-labels">
                    <div class="label-val home">
                        <span>${getAbbr(match.home_team)}</span>
                        <span class="prob-pct">${pHome}%</span>
                    </div>
                    <div class="label-val draw">
                        <span>DRAW</span>
                        <span class="prob-pct">${pDraw}%</span>
                    </div>
                    <div class="label-val away">
                        <span>${getAbbr(match.away_team)}</span>
                        <span class="prob-pct">${pAway}%</span>
                    </div>
                </div>
                ${extraProbText}
            `;
            
            let pickColText = '';
            if (match.stage !== 'Group Stage' && match.advance_method && match.advance_method !== 'Regulation') {
                pickColText = 'DRAW';
                let shortMethod = match.advance_method === 'Extra Time' ? 'AET' : 'Pens';
                let advancer = match.prediction === 'H' ? match.home_team : match.away_team;
                pickColText += `<br><span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-sans); display:block; margin-top:2px;">${getAbbr(advancer)} advances (${shortMethod})</span>`;
            } else {
                if (match.prediction === 'H') pickColText = match.home_team;
                else if (match.prediction === 'D') pickColText = 'DRAW';
                else if (match.prediction === 'A') pickColText = match.away_team;
            }

            let isDraw = pickColText.startsWith('DRAW');
            pickHtml = `
                <span class="tag ${!isDraw ? 'strong' : ''}">${pickColText}</span>
            `;
        }

        let actualScoreHtml = '';
        let actualResultColText = '-';
        if ((match.status === 'FINISHED' || match.status === 'IN_PLAY') && match.actual_home_score >= 0 && match.actual_away_score >= 0) {
            let isCorrect = match.prediction === match.actual_outcome ? '✅ Correct' : '❌ Incorrect';
            actualResultColText = `
                <div style="display:flex; justify-content:center; align-items:center; gap:0.5rem;">
                    <span class="team-abbr" style="color:var(--text-muted); font-size:0.85rem; font-weight:600;">${getAbbr(match.home_team)}</span>
                    <span><strong>${match.actual_home_score} - ${match.actual_away_score}</strong></span>
                    <span class="team-abbr" style="color:var(--text-muted); font-size:0.85rem; font-weight:600;">${getAbbr(match.away_team)}</span>
                </div>
            `;
            let homeScorersHtml = '';
            if (match.home_scorers) {
                homeScorersHtml = match.home_scorers.split(', ').map(s => `<div style="margin-top: 0.2rem;"><span style="font-family: var(--font-hand); color: var(--text-ink); font-size: 1.2rem;">${s}</span> ⚽</div>`).join('');
            }
            let awayScorersHtml = '';
            if (match.away_scorers) {
                awayScorersHtml = match.away_scorers.split(', ').map(s => `<div style="margin-top: 0.2rem;">⚽ <span style="font-family: var(--font-hand); color: var(--text-ink); font-size: 1.2rem;">${s}</span></div>`).join('');
            }
            let shootoutHtml = "";
            let outcomeText = "Full Time Result";
            if (match.home_pen_score !== undefined && match.home_pen_score !== null) {
                outcomeText = "Result After Penalties";
                shootoutHtml = `<div style="font-size: 1.2rem; color: var(--accent-color); margin-top: 0.8rem; text-align: center; font-weight: bold; font-family: var(--font-hand);">(${match.home_pen_score} - ${match.away_pen_score} on penalties)</div>`;
            } else if (match.is_aet) {
                outcomeText = "Result After Extra Time";
            }

            actualScoreHtml = `
                <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px dashed var(--border-color);">
                    <h4 style="color: var(--text-muted); text-align: center; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">${outcomeText}</h4>
                    <div class="full-time-flex">
                        <div class="full-time-team home">
                            <div style="font-weight: bold; font-size: 1.3rem; color: var(--accent-color); margin-bottom: 0.5rem;">${match.home_team} ${match.actual_home_score}</div>
                            ${homeScorersHtml}
                        </div>
                        <div class="full-time-divider">-</div>
                        <div class="full-time-team away">
                            <div style="font-weight: bold; font-size: 1.3rem; color: var(--accent-color); margin-bottom: 0.5rem;">${match.actual_away_score} ${match.away_team}</div>
                            ${awayScorersHtml}
                        </div>
                    </div>
                    ${shootoutHtml}
                </div>
            `;
        }

        const fixtureColText = `
            <div class="match-fixture-wrapper">
                <div class="team-name home">${match.home_team}${homeFlag ? `<img src="${homeFlag}" class="flag" style="width:20px;">` : ''}</div>
                <span class="vs">vs</span>
                <div class="team-name away">${awayFlag ? `<img src="${awayFlag}" class="flag" style="width:20px;">` : ''}${match.away_team}</div>
            </div>
        `;

        tr.innerHTML = `
            <td class="col-date">${dateStr}</td>
            <td class="col-match">${fixtureColText}</td>
            <td class="col-prob">${probHtml}</td>
            <td class="col-prediction">${pickHtml}</td>
            <td class="col-result" style="text-align: center;">${actualResultColText}</td>
        `;

        let exactScoreHtml = '';
        if (hasPlaceholders) {
            exactScoreHtml = `<p class="score-pred" style="color: var(--text-muted); font-size: 0.9rem;">Awaiting teams to finalize...</p>`;
        } else {
            let advanceNote = '';
            let extraProbsHtml = '';
            if (match.stage !== 'Group Stage' && match.advance_method) {
                const winner = match.prediction === 'H' ? match.home_team : (match.prediction === 'A' ? match.away_team : 'Winner');
                const advanceText = match.advance_method === 'Regulation' ? 'in <strong>Regulation Time</strong>' : `via <strong>${match.advance_method}</strong>`;
                advanceNote = `<p style="font-family: var(--font-hand); font-size: 1.4rem; color: var(--accent-color); margin-top: 0.5rem; border-top: 1px dashed var(--border-color); padding-top: 0.5rem;">Tough Match! <strong>${winner}</strong> is predicted to advance ${advanceText}.</p>`;
                
                if (match.advance_method !== 'Regulation' && match.aet_home_prob) {
                    let aetH = (match.aet_home_prob * 100).toFixed(1);
                    let aetA = (match.aet_away_prob * 100).toFixed(1);
                    let penH = (match.pen_home_prob * 100).toFixed(1);
                    let penA = (match.pen_away_prob * 100).toFixed(1);
                    
                    extraProbsHtml = `
                        <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(0,0,0,0.02); border-left: 3px solid var(--accent-color);">
                            <h4 style="font-size: 1.3rem; margin-bottom: 0.5rem;">Extra Time & Penalties Forecast</h4>
                            <div class="extra-probs-flex">
                                <div>
                                    <strong>${getAbbr(match.home_team)}</strong><br>
                                    AET: ${aetH}%<br>
                                    Pens: ${penH}%
                                </div>
                                <div>
                                    <strong>${getAbbr(match.away_team)}</strong><br>
                                    AET: ${aetA}%<br>
                                    Pens: ${penA}%
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
            
            let aetPenScoresHtml = '';
            if (match.stage !== 'Group Stage' && match.advance_method) {
                if (match.advance_method === 'Extra Time' && match.pred_aet_home_score !== undefined && match.pred_aet_home_score !== null) {
                    aetPenScoresHtml = `<br><span style="color:var(--accent-color); font-size: 1.1rem;">After Extra Time: <strong>${match.pred_aet_home_score} - ${match.pred_aet_away_score}</strong></span>`;
                } else if (match.advance_method === 'Penalties' && match.pred_pen_home_score !== undefined && match.pred_pen_home_score !== null) {
                    aetPenScoresHtml = `<br><span style="color:var(--accent-color); font-size: 1.1rem;">Penalties: <strong>${match.pred_pen_home_score} - ${match.pred_pen_away_score}</strong></span>`;
                }
            }
            
            exactScoreHtml = `
                <p class="score-pred">
                    <strong>${match.home_team} ${match.pred_home_score} - ${match.pred_away_score} ${match.away_team}</strong>
                    ${aetPenScoresHtml}
                    <br>
                    <small style="font-family: var(--font-sans);">Calculated via Poisson Regression on historical goal distributions.</small>
                </p>
                ${advanceNote}
                ${extraProbsHtml}
                ${actualScoreHtml}
            `;
        }

        const detailsTr = document.createElement('tr');
        detailsTr.className = "details-row hidden";
        
        if (hasPlaceholders) {
            detailsTr.innerHTML = `
                <td colspan="5">
                    <div class="details-content" style="justify-content: center; padding: 2rem; text-align: center;">
                        <p style="color: var(--text-muted); font-style: italic;">Awaiting teams to finalize for this fixture.</p>
                    </div>
                </td>
            `;
        } else {
            detailsTr.innerHTML = `
                <td colspan="5">
                    <div class="details-content">
                        <div class="details-column">
                            <h4>Exact Score Prediction</h4>
                            ${exactScoreHtml}
                        </div>
                        <div class="details-column rosters-container" id="rosters-${match.id}">
                            <h4>Live Match Details</h4>
                            <p>Loading lineups from ESPN...</p>
                        </div>
                    </div>
                </td>
            `;
        }

        tr.addEventListener('click', () => {
            const isHidden = detailsTr.classList.contains('hidden');
            document.querySelectorAll('.details-row').forEach(r => r.classList.add('hidden'));
            document.querySelectorAll('.match-row').forEach(r => r.classList.remove('active-row'));
            if (isHidden) {
                detailsTr.classList.remove('hidden');
                tr.classList.add('active-row');
                document.body.classList.add('mobile-expanded-mode');
                fetchMatchDetails(match.id);
            } else {
                document.body.classList.remove('mobile-expanded-mode');
            }
        });

        tbody.appendChild(tr);
        tbody.appendChild(detailsTr);
    });
}

async function fetchMatchDetails(matchId) {
    const container = document.getElementById(`rosters-${matchId}`);
    if (container.dataset.loaded) return;

    try {
        const resp = await fetch(`/api/match/${matchId}/summary`);
        const data = await resp.json();
        
        let infoHtml = "";
        
        if (data.rosters && data.rosters.length > 0) {
            infoHtml += `<div style="display:flex; flex-direction: column; gap: 1rem;">`;
            data.rosters.forEach(r => {
                const teamName = r.team ? r.team.displayName : "Team";
                let playersText = "Roster not announced yet.";
                if (r.roster && r.roster.length > 0) {
                     let sortedPlayers = r.roster.slice().sort((a, b) => {
                         const getStat = (p, statName) => {
                             if (!p.stats) return 0;
                             const s = p.stats.find(st => st.name === statName);
                             return s ? s.value : 0;
                         };
                         const aGoals = getStat(a, 'totalGoals');
                         const bGoals = getStat(b, 'totalGoals');
                         if (aGoals !== bGoals) return bGoals - aGoals;
                         
                         const aShots = getStat(a, 'totalShots');
                         const bShots = getStat(b, 'totalShots');
                         if (aShots !== bShots) return bShots - aShots;
                         
                         const aAssists = getStat(a, 'goalAssists');
                         const bAssists = getStat(b, 'goalAssists');
                         return bAssists - aAssists;
                     });
                     
                     playersText = sortedPlayers.slice(0, 3).map(p => p.athlete.displayName).join(", ");
                }
                infoHtml += `<div><strong>${teamName} Top Performers:</strong><br><small>${playersText}</small></div>`;
            });
            infoHtml += `</div>`;
        } else {
            infoHtml = `<p><em>Lineups not announced yet. Live rosters usually available 1hr before kickoff.</em></p>`;
        }

        container.innerHTML = infoHtml;
        container.dataset.loaded = "true";

    } catch (e) {
        container.innerHTML = `<p>Unable to load live details.</p>`;
    }
}
