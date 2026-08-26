import os
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from src.database.db_manager import DatabaseManager

try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    STATSBOMB_AVAILABLE = False

def get_available_competitions() -> pd.DataFrame:
    """List all available StatsBomb open competitions."""
    if not STATSBOMB_AVAILABLE:
        print("Warning: statsbombpy is not installed. Cannot retrieve competitions.")
        return pd.DataFrame()
    try:
        return sb.competitions()
    except Exception as e:
        print(f"Error fetching StatsBomb competitions: {e}")
        return pd.DataFrame()

def get_international_competitions() -> List[Tuple[int, int, str, str]]:
    """Get list of international competitions (World Cup, Euros, Copa America)."""
    df = get_available_competitions()
    if df.empty:
        return []
    
    # Filter for international tournaments
    international_names = ['FIFA World Cup', 'UEFA Euro', 'Copa América']
    filtered = df[df['competition_name'].isin(international_names)]
    
    comps = []
    for _, row in filtered.iterrows():
        comps.append((
            int(row['competition_id']),
            int(row['season_id']),
            row['competition_name'],
            row['season_name']
        ))
    return comps

def load_match_statistics(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch matches and aggregate shot/possession/xG statistics per team."""
    if not STATSBOMB_AVAILABLE:
        return pd.DataFrame()
        
    try:
        # Get matches
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        if matches.empty:
            return pd.DataFrame()
            
        stats_list = []
        for _, match in matches.iterrows():
            match_id = int(match['match_id'])
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Fetch events for the match
            try:
                events = sb.events(match_id=match_id, split=True, flatten_attrs=False)
            except Exception as ev_err:
                print(f"Error fetching events for match {match_id}: {ev_err}")
                continue
                
            shots_df = events.get('shots', pd.DataFrame())
            
            # Calculate shots and xG for Home and Away
            home_shots = 0
            away_shots = 0
            home_xg = 0.0
            away_xg = 0.0
            
            if not shots_df.empty:
                # Home team shots
                home_shots_mask = shots_df['team'] == home_team
                home_shots = int(home_shots_mask.sum())
                
                # Away team shots
                away_shots_mask = shots_df['team'] == away_team
                away_shots = int(away_shots_mask.sum())
                
                # Sum StatsBomb xG (stored inside nested 'shot' attribute if flatten_attrs=False,
                # or as 'shot_statsbomb_xg' if flattened. Let's try to get it safely)
                for _, shot in shots_df.iterrows():
                    shot_team = shot['team']
                    
                    # Try to extract xG
                    xg_val = 0.0
                    shot_details = shot.get('shot')
                    if isinstance(shot_details, dict):
                        xg_val = float(shot_details.get('statsbomb_xg', 0.0))
                    elif 'shot_statsbomb_xg' in shot:
                        xg_val = float(shot['shot_statsbomb_xg']) if not pd.isna(shot['shot_statsbomb_xg']) else 0.0
                        
                    if shot_team == home_team:
                        home_xg += xg_val
                    elif shot_team == away_team:
                        away_xg += xg_val
            
            # Simple possession proxy based on total pass counts
            passes_df = events.get('passes', pd.DataFrame())
            home_possession = 50.0
            away_possession = 50.0
            
            if not passes_df.empty:
                home_passes = (passes_df['team'] == home_team).sum()
                away_passes = (passes_df['team'] == away_team).sum()
                total_passes = home_passes + away_passes
                if total_passes > 0:
                    home_possession = float(home_passes / total_passes * 100)
                    away_possession = float(away_passes / total_passes * 100)
            
            stats_list.append({
                'match_id': match_id,
                'team': home_team,
                'shots': home_shots,
                'possession': home_possession,
                'xg': home_xg
            })
            stats_list.append({
                'match_id': match_id,
                'team': away_team,
                'shots': away_shots,
                'possession': away_possession,
                'xg': away_xg
            })
            
        return pd.DataFrame(stats_list)
    except Exception as e:
        print(f"Error loading match statistics: {e}")
        return pd.DataFrame()

def enrich_database(db_manager: DatabaseManager):
    """Enrich database with StatsBomb match statistics."""
    if not STATSBOMB_AVAILABLE:
        print("StatsBomb library (statsbombpy) not available. Skipping enrichment.")
        return
        
    print("Checking for international tournaments in StatsBomb open data...")
    comps = get_international_competitions()
    if not comps:
        print("No international competitions found in StatsBomb.")
        return
        
    session = db_manager.get_session()
    try:
        from src.database.models import RawTeamStatistic
        
        # Check if we already have statistics
        existing_count = session.query(RawTeamStatistic).count()
        if existing_count > 0:
            print(f"raw_team_statistics table already has {existing_count} rows. Skipping StatsBomb loader.")
            return
            
        for comp_id, season_id, name, season in comps:
            print(f"Fetching match stats for {name} ({season})...")
            stats_df = load_match_statistics(comp_id, season_id)
            
            if not stats_df.empty:
                records = []
                for _, row in stats_df.iterrows():
                    records.append(RawTeamStatistic(
                        match_id=int(row['match_id']),
                        team=row['team'],
                        shots=int(row['shots']),
                        possession=float(row['possession']),
                        xg=float(row['xg'])
                    ))
                session.bulk_save_objects(records)
                session.commit()
                print(f"Saved stats for {len(records)} team-match entries.")
    except Exception as e:
        session.rollback()
        print(f"Error during database enrichment: {e}")
    finally:
        session.close()

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    db_manager = DatabaseManager()
    enrich_database(db_manager)
