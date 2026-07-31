"""
Application d'Analyse Statistiques MLB (via statsapi)
===================================================================
Application Streamlit pour analyser les runs, les sluggers récurrents et les tendances W/L
Données MLB récupérées uniquement via statsapi (API officielle MLB).

Auteur: Généré via MAMMOUTH AI (version pure statsapi)
"""

# ============================================================
# 1. IMPORTS - On importe les bibliothèques nécessaires
# ============================================================
import streamlit as st          # Framework pour créer l'interface web
import pandas as pd             # Manipulation de données (tableaux)
import altair as alt            # Graphiques avancés (ligne de moyenne annotée)
import re                       # Extraction des noms/runs dans les cellules "Joueurs (Runs)"
import time                     # Délais/backoff entre les appels API
from datetime import datetime   # Gestion des dates
from zoneinfo import ZoneInfo   # Gestion des fuseaux horaires (heure US <-> heure française)

import statsapi                 # Utilisation de l'API MLB

# Fuseaux horaires utilisés pour l'affichage double fuseau des heures de match :
# - TZ_US_EASTERN : heure de référence MLB (Heure de l'Est US). `ZoneInfo` bascule
#   automatiquement entre EST (hiver, UTC-5) et EDT (été, UTC-4) selon la date.
# - TZ_PARIS : heure française, pour savoir à quelle heure (locale) suivre le match.
TZ_US_EASTERN = ZoneInfo("America/New_York")
TZ_PARIS = ZoneInfo("Europe/Paris")

# Année MLB courante, basée sur la date du jour aux USA (heure de l'Est), pas sur
# l'heure du serveur/de l'utilisateur : la saison MLB est définie en heure US, donc
# utiliser une autre référence pourrait décaler l'année d'un jour près du Nouvel An.
ANNEE_COURANTE = datetime.now(TZ_US_EASTERN).year


def appeler_avec_retry(fonction, *args, tentatives: int = 3, delai_base: float = 0.5, **kwargs):
    """
    Exécute `fonction(*args, **kwargs)` avec un système de retry + backoff exponentiel.

    Objectif : éviter que l'API MLB (statsapi) fasse "disparaître" silencieusement des
    équipes/joueurs/matchs à cause d'une erreur réseau transitoire ou d'un rejet
    temporaire (timeout, erreur 429/5xx, etc.). Sans cela, le code précédent utilisait
    des `except: continue` qui avalaient l'erreur et sautaient l'équipe/le match sans
    aucune nouvelle tentative ni message - c'est une des causes du bug "certaines
    équipes ne se mettent pas à jour".

    Le délai n'intervient qu'EN CAS D'ÉCHEC (pas avant chaque appel), donc les appels
    réussis (le cas normal) ne sont pas ralentis. Comme la plupart des fonctions qui
    utilisent cet appel sont elles-mêmes mises en cache par Streamlit, ce délai ne
    s'applique de toute façon qu'au premier chargement (cache miss), pas aux reruns.
    """
    derniere_erreur = None
    for tentative in range(1, tentatives + 1):
        try:
            return fonction(*args, **kwargs)
        except Exception as e:
            derniere_erreur = e
            if tentative < tentatives:
                time.sleep(delai_base * (2 ** (tentative - 1)))  # 0.5s, 1s, 2s, ...
    raise derniere_erreur

# ============================================================
# 2. CONFIGURATION DE LA PAGE - Paramètres de l'application
# ============================================================
st.set_page_config(
    page_title="Analyse MLB - Runs & Sluggers",
    page_icon="⚾",
    layout="wide"
)

# ============================================================
# 3. FONCTIONS DE CHARGEMENT DES DONNÉES (avec mise en cache)
# ============================================================

@st.cache_data
def get_teams_mlb_this_year(year: int = None):
    """
    Récupère la liste des 30 équipes MLB pour l'année donnée (par défaut année courante).
    Cet endpoint n'est pas paginé (il retourne toujours les 30 équipes en un seul appel),
    mais on protège quand même l'appel avec un retry en cas d'échec réseau transitoire.
    """
    if year is None:
        year = ANNEE_COURANTE
    season_teams = appeler_avec_retry(statsapi.get, 'teams', {'sportIds': 1, 'season': year})
    data = {}
    for e in season_teams['teams']:
        data[e['abbreviation']] = e['name']
    return data

def extraire_abreviation_equipe(nom_equipe: str) -> str:
    """
    Extrait l'abréviation MLB depuis une chaîne 'ABC - Nom complet'.
    Exemple: 'ARI - Arizona Diamondbacks' -> 'ARI'
    """
    return nom_equipe.split(' - ')[0].strip()

@st.cache_data
def get_team_ids_dict(year: int = None):
    if year is None:
        year = ANNEE_COURANTE
    result = {}
    reponse = appeler_avec_retry(statsapi.get, 'teams', {'sportIds': 1, 'season': year})
    for t in reponse['teams']:
        result[t['abbreviation']] = t['id']
    return result

@st.cache_data
def charger_donnees_equipe(annee: int = None, equipe_abbr: str = None) -> pd.DataFrame:
    """
    Charge les données de match pour une équipe donnée (statsapi)
    Affiche deux colonnes distinctes: 'Équipe Domicile' et 'Équipe Extérieur'.
    """
    if annee is None:
        annee = ANNEE_COURANTE
    if equipe_abbr is None:
        return pd.DataFrame()

    team_ids = get_team_ids_dict(annee)
    team_id = team_ids.get(equipe_abbr)
    if not team_id:
        return pd.DataFrame()
    nom_equipe_select = statsapi.lookup_team(team_id)[0]['name'] if team_id else ""

    try:
        schedule = statsapi.schedule(team=team_id, start_date=f"{annee}-03-01", end_date=f"{annee}-11-30")
        matchs = []
        for g in schedule:
            if g.get('status', '') != "Final":
                continue

            home_team = g.get('home_name') or (
                g.get('teams', {}).get('home', {}).get('team', {}).get('name')
            )
            away_team = g.get('away_name') or (
                g.get('teams', {}).get('away', {}).get('team', {}).get('name')
            )

            home_score = g.get('home_score')
            away_score = g.get('away_score')

            if home_score is None or away_score is None:
                continue
            try:
                home_score = int(home_score)
                away_score = int(away_score)
            except Exception:
                continue

            nom_home = (home_team or "").strip().casefold()
            nom_away = (away_team or "").strip().casefold()
            nom_equipe_ref = (nom_equipe_select or "").strip().casefold()

            est_ext = False
            est_dom = False

            home_away_val = None
            if 'home_away' in g:
                home_away_val = g['home_away']
            elif 'Home_Away' in g:
                home_away_val = g['Home_Away']
            elif '@' in g:
                home_away_val = g['@']

            if home_away_val is not None:
                if str(home_away_val).strip().lower() in ['away', '@', 'x']:
                    est_ext = True
                else:
                    est_dom = True
            else:
                if nom_equipe_ref == nom_away:
                    est_ext = True
                elif nom_equipe_ref == nom_home:
                    est_dom = True
                else:
                    # fallback minimal
                    if nom_equipe_ref in (nom_away or ''):
                        est_ext = True
                    elif nom_equipe_ref in (nom_home or ''):
                        est_dom = True
                    else:
                        continue

            if not (est_ext or est_dom):
                continue

            if est_ext:
                equipe_domicile = home_team
                equipe_exterieur = nom_equipe_select
                runs = away_score
                runs_adverses = home_score
            elif est_dom:
                equipe_domicile = nom_equipe_select
                equipe_exterieur = away_team
                runs = home_score
                runs_adverses = away_score
            else:
                continue

            if runs > runs_adverses:
                wl = "W"
            elif runs < runs_adverses:
                wl = "L"
            else:
                wl = "T"

            matchs.append({
                "Date": g['game_date'][:10] if 'game_date' in g else '',
                "Équipe Domicile": equipe_domicile,
                "Équipe Extérieur": equipe_exterieur,
                "R": runs,
                "RA": runs_adverses,
                "W/L": wl,
                "game_id": g.get('game_id'),
                "Est_Domicile": est_dom
            })
        df = pd.DataFrame(matchs)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données pour {equipe_abbr} ({annee}): {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_stats_offensives_match(game_id: int, est_domicile: bool):
    """
    Récupère, via le boxscore statsapi d'un match, les runs ET les home runs marqués
    par chaque joueur de l'équipe (domicile ou extérieur) lors de ce match (un seul
    appel au boxscore, pour éviter de doubler les requêtes réseau).
    Retourne une liste de dicts {'name': str, 'runs': int, 'hr': int}.
    """
    if not game_id:
        return []
    try:
        box = appeler_avec_retry(statsapi.boxscore_data, int(game_id))
        batters = box.get('homeBatters', []) if est_domicile else box.get('awayBatters', [])

        stats_par_joueur = {}
        for b in batters:
            if not b.get('personId'):
                continue  # ligne d'en-tête du tableau, pas un joueur
            try:
                runs = int(b.get('r', 0) or 0)
            except (ValueError, TypeError):
                runs = 0
            try:
                hr = int(b.get('hr', 0) or 0)
            except (ValueError, TypeError):
                hr = 0
            if runs > 0 or hr > 0:
                nom = b.get('name', 'Inconnu')
                if nom not in stats_par_joueur:
                    stats_par_joueur[nom] = {'runs': 0, 'hr': 0}
                stats_par_joueur[nom]['runs'] += runs
                stats_par_joueur[nom]['hr'] += hr

        return [{'name': nom, 'runs': s['runs'], 'hr': s['hr']} for nom, s in stats_par_joueur.items()]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_matchs_avec_scoreurs(annee: int, equipe_abbr: str):
    """
    Enrichit les données de match avec la liste des scoreurs de runs ET de home runs
    par match, et calcule le cumul de runs / home runs marqués par joueur sur toute
    la période chargée.
    Retourne (df_matchs_enrichi, df_meilleurs_scoreurs_runs, df_meilleurs_scoreurs_hr).
    """
    df = charger_donnees_equipe(annee, equipe_abbr)
    if df.empty or 'game_id' not in df.columns:
        return df, pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    colonne_joueurs_runs = []
    colonne_joueurs_hr = []
    colonne_stats_brutes = []
    cumul_runs = {}
    cumul_hr = {}

    for _, ligne in df.iterrows():
        stats_batteurs = get_stats_offensives_match(ligne['game_id'], bool(ligne['Est_Domicile']))

        # On conserve les données BRUTES (liste de dicts {name, runs, hr}) dans une
        # colonne cachée, en plus de la version texte formatée pour l'affichage.
        # Toute agrégation ultérieure (ex: résumé des 10 derniers matchs) doit
        # additionner ces valeurs brutes directement, et NE PLUS reparser le texte
        # formaté ci-dessous : reparser une chaîne comme "Pederson (2), Burger (1)"
        # est fragile (virgules dans les noms au format "Nom, Initiale", suffixe de
        # désambiguïsation qui peut varier d'un match à l'autre pour un même
        # joueur, etc.) et peut produire des totaux différents du contenu réel du
        # tableau. Garder les données brutes garantit que les totaux affichés
        # ailleurs correspondent TOUJOURS exactement à ce tableau.
        colonne_stats_brutes.append(stats_batteurs)

        # Chaque cellule liste "Nom (valeur)" par joueur, séparés par des virgules
        entrees_runs = [f"{s['name']} ({s['runs']})" for s in stats_batteurs if s['runs'] > 0]
        colonne_joueurs_runs.append(", ".join(entrees_runs) if entrees_runs else "—")

        entrees_hr = [f"{s['name']} ({s['hr']})" for s in stats_batteurs if s['hr'] > 0]
        colonne_joueurs_hr.append(", ".join(entrees_hr) if entrees_hr else "—")

        for s in stats_batteurs:
            if s['runs'] > 0:
                cumul_runs[s['name']] = cumul_runs.get(s['name'], 0) + s['runs']
            if s['hr'] > 0:
                cumul_hr[s['name']] = cumul_hr.get(s['name'], 0) + s['hr']

    df['Joueurs (Runs)'] = colonne_joueurs_runs
    df['Joueurs (HR)'] = colonne_joueurs_hr
    df['_offensive_stats'] = colonne_stats_brutes  # colonne interne (non affichée) : liste de dicts {name, runs, hr}

    df_meilleurs_runs = pd.DataFrame(
        [{'Joueur': nom, 'Runs Marqués': total} for nom, total in cumul_runs.items()]
    )
    if not df_meilleurs_runs.empty:
        df_meilleurs_runs = df_meilleurs_runs.sort_values('Runs Marqués', ascending=False).reset_index(drop=True)

    df_meilleurs_hr = pd.DataFrame(
        [{'Joueur': nom, 'Home Runs': total} for nom, total in cumul_hr.items()]
    )
    if not df_meilleurs_hr.empty:
        df_meilleurs_hr = df_meilleurs_hr.sort_values('Home Runs', ascending=False).reset_index(drop=True)

    return df, df_meilleurs_runs, df_meilleurs_hr


def parser_cellule_joueurs(cellule: str) -> dict:
    """
    Parse une cellule du type "Nom (N), Nom2 (N2), ..." et retourne un dict {nom: total}.

    Une cellule peut contenir plusieurs joueurs séparés par des virgules, ex:
    "Freeman, F (2), Muncy (1), Hernández, T (1)". Certains noms MLB contiennent
    eux-mêmes une virgule (format "Nom, Initiale"), donc on ne peut pas simplement
    découper sur toutes les virgules : on découpe plutôt sur chaque entrée complète
    "... (N)" (recherche non-gourmande jusqu'à la prochaine parenthèse de valeur).
    """
    cumul = {}
    if not cellule or cellule == "—":
        return cumul

    entrees = re.findall(r'(.+?\(\d+\))(?:,\s*|$)', cellule)
    for entree in entrees:
        entree = entree.strip()
        if not entree:
            continue
        correspondance = re.match(r'^(.*)\((\d+)\)$', entree)
        if correspondance:
            nom = correspondance.group(1).strip()
            valeur = int(correspondance.group(2))
        else:
            nom = entree
            valeur = 1
        cumul[nom] = cumul.get(nom, 0) + valeur

    return cumul


def calculer_resume_10_derniers_matchs(df_derniers: pd.DataFrame):
    """
    À partir des données des 10 derniers matchs, calcule, pour les runs ET pour les
    home runs : la moyenne marquée sur ces matchs, le cumul EXACT par joueur, et le
    top 3 des joueurs les plus récurrents.

    --- CORRECTIF (totaux par joueur incorrects) ---
    Auparavant, cette fonction reparsait les colonnes texte déjà formatées
    ('Joueurs (Runs)' / 'Joueurs (HR)', ex: "Pederson (2), Burger (1)") avec une
    regex pour reconstituer les totaux. Cette approche est fragile - un même nom
    peut apparaître avec un suffixe de désambiguïsation différent d'un match à
    l'autre ("Duran" vs "Duran, E"), ce qui pouvait faire diverger silencieusement
    la somme calculée du contenu réel du tableau affiché.

    La fonction additionne maintenant DIRECTEMENT les statistiques brutes par match
    (colonne interne '_offensive_stats', une liste de dicts {name, runs, hr} par
    match - la même source que celle utilisée pour construire les colonnes
    affichées), sans repasser par aucun texte formaté. Le total obtenu correspond
    donc toujours exactement à la somme des valeurs visibles dans le tableau des
    10 derniers matchs. Le parsing par regex (`parser_cellule_joueurs`) n'est
    conservé qu'en repli, si jamais la colonne brute n'est pas disponible.

    Retourne (moyenne_runs, top3_runs, moyenne_hr, top3_hr, cumul_runs, cumul_hr) :
      - top3_* est une liste de tuples (nom, total) limitée aux 3 plus hauts totaux.
      - cumul_runs / cumul_hr sont les dictionnaires COMPLETS {nom: total} (non
        tronqués), à utiliser dès qu'on a besoin du total exact d'un joueur qui
        n'est pas forcément dans le top 3 de l'AUTRE catégorie.
    """
    if df_derniers.empty or 'R' not in df_derniers.columns:
        return None, [], None, [], {}, {}

    moyenne_runs = pd.to_numeric(df_derniers['R'], errors='coerce').mean()

    a_stats_brutes = '_offensive_stats' in df_derniers.columns
    a_colonne_hr = 'Joueurs (HR)' in df_derniers.columns

    cumul_runs = {}
    cumul_hr = {}

    if a_stats_brutes:
        for stats_match in df_derniers['_offensive_stats']:
            for s in (stats_match or []):
                if s.get('runs', 0) > 0:
                    cumul_runs[s['name']] = cumul_runs.get(s['name'], 0) + s['runs']
                if s.get('hr', 0) > 0:
                    cumul_hr[s['name']] = cumul_hr.get(s['name'], 0) + s['hr']
    else:
        # Repli (rétro-compatibilité) : si la colonne brute n'existe pas, on
        # retombe sur le parsing texte, moins fiable mais fonctionnel.
        for cellule in df_derniers.get('Joueurs (Runs)', []):
            for nom, valeur in parser_cellule_joueurs(cellule).items():
                cumul_runs[nom] = cumul_runs.get(nom, 0) + valeur
        if a_colonne_hr:
            for cellule in df_derniers['Joueurs (HR)']:
                for nom, valeur in parser_cellule_joueurs(cellule).items():
                    cumul_hr[nom] = cumul_hr.get(nom, 0) + valeur

    top3_runs = sorted(cumul_runs.items(), key=lambda x: x[1], reverse=True)[:3]

    moyenne_hr = None
    top3_hr = []
    if a_stats_brutes or a_colonne_hr:
        nb_matchs = len(df_derniers)
        moyenne_hr = (sum(cumul_hr.values()) / nb_matchs) if nb_matchs else 0.0
        top3_hr = sorted(cumul_hr.items(), key=lambda x: x[1], reverse=True)[:3]

    return moyenne_runs, top3_runs, moyenne_hr, top3_hr, cumul_runs, cumul_hr


@st.cache_data(show_spinner=False, ttl=300)
def obtenir_match_du_jour(team_id: int):
    """
    Cherche, dans le calendrier MLB du jour (date du jour AUX USA, heure de l'Est),
    un match impliquant l'équipe donnée. Retourne un dict avec l'adversaire, le
    statut domicile/extérieur, les lanceurs partants prévus (des deux côtés), le
    stade, ainsi que l'heure du match dans DEUX fuseaux horaires (US Eastern ET
    France), ou None si aucun match n'est prévu aujourd'hui (heure US) pour cette
    équipe.

    --- CORRECTIF (jour de référence) ---
    On détermine "aujourd'hui" avec `datetime.now(TZ_US_EASTERN)` et non plus avec
    l'heure système brute. La saison MLB (et l'endpoint `schedule` de statsapi) est
    organisée selon le jour calendaire AMÉRICAIN : pour un utilisateur en France,
    l'heure locale peut déjà être "demain" (ex: 1h du matin en France = encore
    19h la veille sur la côte Est) - utiliser l'heure locale du serveur/utilisateur
    aurait pu faire chercher le mauvais jour de match, ou n'en trouver aucun alors
    qu'un match est bien prévu ce soir (heure US).

    Le cache utilise un `ttl=300` (5 minutes) - contrairement aux autres fonctions
    de chargement, ces données (lanceur probable, statut du match) peuvent changer
    en cours de journée, donc on ne les garde pas en cache indéfiniment.
    """
    if not team_id:
        return None
    aujourdhui_us = datetime.now(TZ_US_EASTERN).strftime('%Y-%m-%d')
    try:
        matchs_du_jour = appeler_avec_retry(statsapi.schedule, date=aujourdhui_us, team=team_id)
    except Exception:
        return None
    if not matchs_du_jour:
        return None

    match = matchs_du_jour[0]  # cas des double programmes (doubleheader) : on retient le 1er match
    est_domicile = (match.get('home_id') == team_id)

    # --- Double fuseau horaire : US Eastern (référence MLB) <-> France ---
    # `game_datetime` est fourni par statsapi en UTC (ex: "2026-07-30T23:10:00Z").
    # On le convertit dans les deux fuseaux demandés. Beaucoup de matchs se jouent
    # en soirée aux USA, donc l'heure française correspondante tombe très souvent
    # le LENDEMAIN matin : on affiche donc systématiquement la date (jour/mois)
    # avec l'heure française pour éviter toute ambiguïté sur le jour civil
    # (ex: "31/07 à 01:00"), même quand elle tombe le même jour que l'heure US.
    heure_us_str = None
    heure_paris_str = None
    game_datetime_str = match.get('game_datetime')
    if game_datetime_str:
        try:
            dt_utc = datetime.fromisoformat(game_datetime_str.replace('Z', '+00:00'))
            dt_us = dt_utc.astimezone(TZ_US_EASTERN)
            dt_paris = dt_utc.astimezone(TZ_PARIS)
            # `%Z` donne l'abréviation correcte (EST ou EDT) selon la date, gérée
            # automatiquement par `zoneinfo` (bascule heure d'été/hiver US).
            heure_us_str = dt_us.strftime('%H:%M %Z')
            heure_paris_str = dt_paris.strftime('%d/%m à %H:%M')
        except Exception:
            heure_us_str = None
            heure_paris_str = None

    return {
        'adversaire': match.get('away_name') if est_domicile else match.get('home_name'),
        'est_domicile': est_domicile,
        'lanceur_notre_equipe': match.get('home_probable_pitcher') if est_domicile else match.get('away_probable_pitcher'),
        'lanceur_adverse': match.get('away_probable_pitcher') if est_domicile else match.get('home_probable_pitcher'),
        'heure_us': heure_us_str or "—",
        'heure_paris': heure_paris_str or "—",
        'statut': match.get('status'),
        'venue': match.get('venue_name'),
    }


@st.cache_data(show_spinner=False)
def obtenir_stats_lanceur(nom_lanceur: str, annee: int):
    """
    Recherche un lanceur par son nom complet (tel que renvoyé par le calendrier du
    jour) et retourne ses statistiques de la saison en cours (ERA, WHIP, runs
    alloués, HR alloués, HR/9, matchs comme titulaire) via statsapi.
    Retourne None si le nom est vide, introuvable, ou si les stats sont absentes
    (ex: lanceur de relève sans départ, débutant sans historique, etc.).
    """
    if not nom_lanceur:
        return None
    try:
        resultats = appeler_avec_retry(statsapi.lookup_player, nom_lanceur)
        if not resultats:
            return None
        player_id = resultats[0]['id']
        stats_joueur = appeler_avec_retry(
            statsapi.player_stat_data, player_id, group="pitching", type="season"
        )
        for bloc in stats_joueur.get('stats', []):
            if bloc.get('type') == 'season' and bloc.get('group') == 'pitching':
                s = bloc.get('stats', {})
                if not s.get('era'):
                    return None
                return {
                    'nom': resultats[0].get('fullName', nom_lanceur),
                    'era': float(s.get('era') or 0),
                    'whip': float(s.get('whip') or 0),
                    'runs_alloues': int(s.get('runs') or 0),
                    'hr_alloues': int(s.get('homeRuns') or 0),
                    'hr_par_9': float(s.get('homeRunsPer9') or 0),
                    'matchs_titulaire': int(s.get('gamesStarted') or s.get('gamesPitched') or 0),
                }
        return None
    except Exception:
        return None


def predire_runs_match(moyenne_runs_equipe, moyenne_ra_equipe, stats_lanceur_adverse):
    """
    Estimation heuristique (PAS un modèle statistique validé) du nombre de runs que
    l'équipe sélectionnée pourrait marquer aujourd'hui, ainsi que du total de runs
    du match, en croisant :
      - la moyenne de runs marqués par l'équipe sur ses 10 derniers matchs,
      - les stats du lanceur partant adverse (ERA, WHIP) - un ERA/WHIP élevé
        indique un lanceur plus "battable", donc on augmente l'estimation,
      - la moyenne de runs concédés par l'équipe sur ses 10 derniers matchs,
        utilisée comme proxy raisonnable de l'attaque adverse (faute de connaître
        le lanceur partant de notre propre équipe, hors périmètre de la demande).
    Retourne un dict {'runs_equipe', 'total_match', 'confiance'} ou None si aucune
    donnée de forme récente n'est disponible pour l'équipe.
    """
    if moyenne_runs_equipe is None:
        return None

    if stats_lanceur_adverse is not None and stats_lanceur_adverse.get('era', 0) > 0:
        era = stats_lanceur_adverse['era']
        whip = stats_lanceur_adverse['whip']
        # Moyenne pondérée entre la forme offensive de l'équipe et la vulnérabilité du lanceur adverse
        runs_estimes_equipe = (moyenne_runs_equipe * 0.55) + (era * 0.45)
        # Un WHIP élevé (plus de coureurs sur les buts) augmente l'estimation, un WHIP très bas la réduit
        if whip >= 1.35:
            runs_estimes_equipe *= 1.12
        elif whip <= 1.05:
            runs_estimes_equipe *= 0.90
        confiance = "Élevée" if stats_lanceur_adverse.get('matchs_titulaire', 0) >= 8 else "Moyenne"
    else:
        # Pas de stats fiables sur le lanceur adverse -> on se base uniquement sur la forme offensive de l'équipe
        runs_estimes_equipe = moyenne_runs_equipe
        confiance = "Faible"

    runs_estimes_adverse = moyenne_ra_equipe if moyenne_ra_equipe is not None and pd.notna(moyenne_ra_equipe) else moyenne_runs_equipe
    total_runs_estime = runs_estimes_equipe + runs_estimes_adverse

    return {
        'runs_equipe': round(runs_estimes_equipe, 1),
        'total_match': round(total_runs_estime, 1),
        'confiance': confiance,
    }


def predire_probabilite_victoire(
    moyenne_runs_nous,
    moyenne_offense_adverse,
    stats_lanceur_nous,
    stats_lanceur_adverse,
    est_domicile: bool,
):
    """
    Estimation heuristique (PAS un modèle statistique validé - aucune régression logistique
    entraînée sur des données historiques ici, juste une pondération "de bon sens") de la
    probabilité de victoire de l'équipe sélectionnée ("nous") face à son adversaire du jour,
    exprimée en pourcentage pour CHAQUE équipe (les deux valeurs retournées somment à 100%).

    --- Port direct de la fonction du même nom dans KBO_Stats_App / NPB_Stats_App ---
    La formule, les pondérations et les constantes ci-dessous sont IDENTIQUES à la version
    KBO/NPB (seule la couche de récupération des données change : ici, `stats_lanceur_nous`
    et `stats_lanceur_adverse` viennent de `obtenir_stats_lanceur()` via statsapi, au lieu
    du scraping KBO/NPB).

    --- Les 3 facteurs retenus, et leur pondération ---
    1. LANCEURS PARTANTS PRÉVUS (poids 60% dans le score combiné - facteur jugé le PLUS
       déterminant : à l'échelle d'UN match de baseball, un lanceur partant influence
       directement 5 à 7 manches sur 9, un poids qu'aucun frappeur isolé n'a à lui seul).
       Pour chaque lanceur, on calcule un "indice de qualité" =
       (1/ERA) * 0.7 + (1/WHIP) * 0.3 : l'ERA pèse plus car c'est la statistique la plus
       lisible/suivie, le WHIP vient l'affiner (il capture aussi les coureurs laissés sur
       les buts, pas seulement les points encaissés). Plus l'indice est élevé (ERA/WHIP
       BAS), plus la probabilité penche vers l'équipe de ce lanceur. La part de chaque
       équipe dans ce facteur est simplement son indice rapporté à la somme des deux
       indices (ex: si notre lanceur a un indice deux fois plus élevé que l'adverse, on
       obtient 2/3 - 1/3, PAS 100% - 0%, pour rester réaliste).
    2. DYNAMIQUE OFFENSIVE RÉCENTE (poids 40% dans le score combiné) : moyenne de runs
       marqués sur les 10 derniers matchs de CHAQUE équipe. Pour notre équipe, on réutilise
       directement `moyenne_runs_10` (déjà calculé ailleurs dans l'onglet). Pour l'attaque
       ADVERSE, faute de recharger séparément ses 10 derniers matchs (appel réseau
       supplémentaire non indispensable dans le temps imparti), on réutilise EXACTEMENT le
       même proxy que `predire_runs_match` juste au-dessus : la moyenne de runs CONCÉDÉS
       par NOTRE équipe sur ses 10 derniers matchs (`moyenne_ra_10`), un indicateur
       indirect mais raisonnable de la force offensive à laquelle notre équipe a été
       récemment confrontée. Ce choix est documenté ici explicitement plutôt que caché.
    3. AVANTAGE DU TERRAIN (bonus fixe de +3 points de pourcentage, PAS un facteur pondéré
       avec les deux précédents - appliqué APRÈS le score combiné) pour l'équipe qui reçoit.
       Valeur choisie par prudence : les études sabermétriques MLB situent le taux de
       victoires à domicile autour de 53-54% en moyenne sur longue période (soit un
       avantage net d'environ 3 à 4 points par rapport à un match parfaitement équilibré à
       50/50) ; on retient ici la borne basse (+3) pour ne pas sur-pondérer un facteur
       secondaire.

    --- Dégradation gracieuse (données manquantes) ---
    - Lanceur sans ERA exploitable (`stats_lanceur_nous`/`stats_lanceur_adverse` vaut None,
      ou n'a pas de champ 'era' renseigné - ex: lanceur de relève sans profil de titulaire
      cette saison) : ce lanceur reçoit un ERA/WHIP "neutres" (`ERA_NEUTRE`/`WHIP_NEUTRE`,
      des moyennes de ligue approximatives), ce qui revient à neutraliser sa contribution
      individuelle SANS jamais planter ni fausser l'estimation vers un 0%/100% trompeur.
      Si les DEUX lanceurs manquent, le facteur 1 devient entièrement neutre (50/50), et
      seuls les facteurs 2 et 3 continuent à jouer.
    - Moyenne de runs manquante (`None`/`NaN`, ex: moins de 10 matchs joués cette saison) :
      remplacée par une moyenne "neutre" (`RUNS_NEUTRE`), pour la même raison.
    - Aucune combinaison de données manquantes ne peut faire planter cette fonction : au
      pire (aucune donnée du tout), elle retombe sur un 50/50 + bonus domicile.

    Retourne un tuple (pct_nous, pct_adverse) de deux flottants arrondis à 1 décimale dont
    la SOMME vaut exactement 100.0, chacun bornée entre 5.0 et 95.0 : une simple heuristique
    ne doit jamais afficher une fausse "certitude absolue" à 0% ou 100%.
    """
    # Valeurs "neutres" de repli (moyennes de ligue approximatives), utilisées uniquement
    # quand une donnée réelle manque, pour neutraliser proprement le facteur concerné.
    # Ces constantes sont identiques à la version KBO (moyennes de baseball professionnel
    # généralistes), et restent des ordres de grandeur raisonnables pour la MLB également.
    ERA_NEUTRE = 4.50    # ERA moyen approximatif toutes équipes confondues
    WHIP_NEUTRE = 1.35   # WHIP moyen approximatif toutes équipes confondues
    RUNS_NEUTRE = 4.50   # Runs/match moyens approximatifs
    BONUS_DOMICILE = 3.0  # Points de pourcentage (voir justification ci-dessus)

    def _indice_qualite_lanceur(stats_lanceur):
        """Indice de qualité d'un lanceur (plus haut = meilleur), avec repli neutre."""
        if stats_lanceur is not None and stats_lanceur.get('era'):
            era = stats_lanceur['era']
            whip = stats_lanceur.get('whip') or WHIP_NEUTRE
        else:
            era, whip = ERA_NEUTRE, WHIP_NEUTRE
        return (1.0 / era) * 0.7 + (1.0 / whip) * 0.3

    # --- Facteur 1 : lanceurs partants (poids 60%) ---
    qualite_nous = _indice_qualite_lanceur(stats_lanceur_nous)
    qualite_adverse = _indice_qualite_lanceur(stats_lanceur_adverse)
    part_lanceurs_nous = qualite_nous / (qualite_nous + qualite_adverse)

    # --- Facteur 2 : dynamique offensive récente (poids 40%) ---
    runs_nous = (
        moyenne_runs_nous if moyenne_runs_nous is not None and pd.notna(moyenne_runs_nous)
        else RUNS_NEUTRE
    )
    runs_adverse = (
        moyenne_offense_adverse if moyenne_offense_adverse is not None and pd.notna(moyenne_offense_adverse)
        else RUNS_NEUTRE
    )
    somme_runs = runs_nous + runs_adverse
    part_offense_nous = (runs_nous / somme_runs) if somme_runs > 0 else 0.5

    # --- Score combiné (facteurs 1 + 2), puis conversion en pourcentage ---
    part_combinee_nous = (part_lanceurs_nous * 0.6) + (part_offense_nous * 0.4)
    pct_nous = part_combinee_nous * 100.0

    # --- Facteur 3 : avantage du terrain (bonus fixe, appliqué après coup) ---
    pct_nous += BONUS_DOMICILE if est_domicile else -BONUS_DOMICILE

    # Bornes de sécurité (jamais 0%/100% avec une simple heuristique) + normalisation
    # stricte à 100% (l'adversaire récupère exactement le complément).
    pct_nous = max(5.0, min(95.0, pct_nous))
    pct_adverse = 100.0 - pct_nous

    return round(pct_nous, 1), round(pct_adverse, 1)


def predire_joueurs_du_jour(cumul_runs_10, cumul_hr_10, stats_lanceur_adverse, top_n: int = 3):
    """
    Construit une liste de joueurs "en forme" et calcule pour chacun un indice de
    confiance (0-100) croisant leur activité récente avec les faiblesses du
    lanceur adverse du jour (ERA, WHIP, HR/9 encaissés).

    --- CORRECTIF (un joueur pouvait afficher "0 run" alors qu'il en avait marqué) ---
    Cette fonction recevait auparavant uniquement les listes TOP 3 (top3_runs_10 /
    top3_hr_10, déjà tronquées à 3 éléments chacune). Un joueur présent dans le
    top 3 des HR mais pas dans le top 3 des runs (car d'autres joueurs avaient
    plus de runs) se voyait donc afficher "0 run" même s'il avait réellement
    marqué plusieurs runs sur les 10 derniers matchs.

    La fonction prend maintenant directement `cumul_runs_10` / `cumul_hr_10`, les
    dictionnaires COMPLETS (non tronqués) de tous les joueurs. On sélectionne les
    candidats "en forme" via le top 3 de chaque catégorie (comme avant), mais on
    va chercher leur total EXACT (runs ET HR) dans ces dictionnaires complets, ce
    qui garantit un affichage fidèle au tableau des 10 derniers matchs.

    Retourne une liste de dicts triée par indice décroissant, limitée à `top_n`.
    """
    cumul_runs_10 = cumul_runs_10 or {}
    cumul_hr_10 = cumul_hr_10 or {}

    if not cumul_runs_10 and not cumul_hr_10:
        return []

    # Candidats "en forme" = présents dans le top 3 d'AU MOINS une des deux
    # catégories (runs ou HR) - mais leur total affiché sera toujours le total
    # RÉEL (les deux dictionnaires complets), jamais une valeur tronquée à 0.
    top3_noms_runs = {nom for nom, _ in sorted(cumul_runs_10.items(), key=lambda x: x[1], reverse=True)[:3]}
    top3_noms_hr = {nom for nom, _ in sorted(cumul_hr_10.items(), key=lambda x: x[1], reverse=True)[:3]}
    candidats = top3_noms_runs | top3_noms_hr

    if not candidats:
        return []

    # Facteur de vulnérabilité du lanceur adverse : plus son ERA/WHIP/HR-par-9 sont
    # élevés, plus il est jugé "battable" (facteur > 1) ; un lanceur dominant réduit
    # le facteur (< 1). Le facteur est borné pour rester réaliste (pas d'emballement).
    facteur_adverse = 1.0
    if stats_lanceur_adverse is not None and stats_lanceur_adverse.get('era', 0) > 0:
        era = stats_lanceur_adverse['era']
        whip = stats_lanceur_adverse['whip']
        hr9 = stats_lanceur_adverse['hr_par_9']
        facteur_adverse += max(0, (era - 4.0)) * 0.08
        facteur_adverse += max(0, (whip - 1.20)) * 0.5
        facteur_adverse += max(0, (hr9 - 1.0)) * 0.15
        facteur_adverse = max(0.7, min(facteur_adverse, 1.6))

    resultats = []
    for nom in candidats:
        runs_10 = cumul_runs_10.get(nom, 0)
        hr_10 = cumul_hr_10.get(nom, 0)
        indice_brut = (runs_10 * 8) + (hr_10 * 20)  # le HR pèse plus car plus rare qu'un run
        indice = min(95, round(indice_brut * facteur_adverse))
        if indice <= 0:
            continue
        if indice >= 65:
            confiance = "Élevée"
        elif indice >= 35:
            confiance = "Moyenne"
        else:
            confiance = "Faible"
        resultats.append({
            'nom': nom,
            'runs_10': runs_10,
            'hr_10': hr_10,
            'indice': indice,
            'confiance': confiance,
        })

    resultats = sorted(resultats, key=lambda x: x['indice'], reverse=True)
    return resultats[:top_n]


# ============================================================
# 4. LISTE DES ÉQUIPES MLB (Abréviations officielles)
# ============================================================

# ============================================================
# 5. INTERFACE PRINCIPALE
# ============================================================

st.title("⚾ Analyse Statistiques MLB")
st.markdown("### Explorez les runs, les prédictions du jour et les tendances W/L")

# Sidebar pour les paramètres globaux
with st.sidebar:
    st.header("⚙️ Paramètres")
    saison_options = list(range(ANNEE_COURANTE, ANNEE_COURANTE-5, -1))
    annee = int(st.selectbox(
        "Sélectionnez la saison:",
        options=saison_options,
        index=0
    ))
    st.markdown("---")
    st.markdown("**Légende des abréviations:**")
    st.markdown("""
    - **R** : Runs (Points marqués)
    - **RA** : Runs Against (Points concédés)
    - **HR** : Home Runs (Coup de circuit)
    - **W** : Wins (Victoires)
    - **L** : Losses (Défaites)
    """)

# Récupération dynamique de la liste des équipes MLB
EQUIPES_MLB = get_teams_mlb_this_year(annee)

# ============================================================
# 6. ONGLETS PRINCIPAUX
# ============================================================
onglets = st.tabs([
    "📊 Analyse par Équipe",
    "🔮 Prédictions du jour"
])

# --------------------------------------------------------------
# ONGLET 1: ANALYSE PAR ÉQUIPE
# --------------------------------------------------------------
with onglets[0]:
    st.header("📊 Analyse des Runs par Équipe")

    col1, col2 = st.columns([1, 3])

    with col1:
        options_equipes = [f"{abbr} - {nom}" for abbr, nom in EQUIPES_MLB.items()]
        equipe_selectionnee = st.selectbox(
            "Choisissez une équipe:",
            options=options_equipes
        )

    equipe_abbr = extraire_abreviation_equipe(equipe_selectionnee)

    # Chargement des données de matchs, enrichies avec les scoreurs de runs et de HR (boxscores statsapi)
    with st.spinner(f"Chargement des données et des boxscores pour les {EQUIPES_MLB[equipe_abbr]} ({annee})... (peut prendre un moment)"):
        df_matchs, df_meilleurs_scoreurs, df_meilleurs_hr = get_matchs_avec_scoreurs(annee, equipe_abbr)

    # Valeurs par défaut du résumé des 10 derniers matchs : elles sont réaffectées plus bas
    # si les données sont disponibles, mais doivent exister dès maintenant car l'onglet
    # "Prédictions du jour" (exécuté après celui-ci) les réutilise.
    moyenne_runs_10, top3_runs_10, moyenne_hr_10, top3_hr_10 = None, [], None, []
    cumul_runs_10, cumul_hr_10 = {}, {}

    st.markdown("---")
    st.subheader("🔝 Classement Home Runs dans l'équipe")

    # -------- NOUVELLE LOGIQUE : Récupérer les Home Runs via statsapi uniquement (plus de pybaseball/FanGraphs) --------
    @st.cache_data(show_spinner=False)
    def get_top_hr_batteurs_statsapi(annee, equipe_abbr, top_n=3):
        """Récupère le top Home Runs via l'API officielle MLB/statsapi, pour une équipe/saison donnée."""
        result = []
        try:
            team_ids = get_team_ids_dict(annee)
            team_id = team_ids.get(equipe_abbr)
            if not team_id:
                return []

            roster_data = appeler_avec_retry(
                statsapi.get, 'team_roster', {'teamId': team_id, 'season': annee}
            )

            for item in roster_data.get('roster', []):
                player_id = item['person']['id']
                nom = item['person']['fullName']

                player_stats = appeler_avec_retry(
                    statsapi.player_stat_data, player_id, group="hitting", type="season"
                )

                home_runs = 0
                for stat_item in player_stats.get('stats', []):
                    if 'stats' in stat_item:
                        home_runs = int(stat_item['stats'].get('homeRuns') or 0)
                        break

                result.append({'name': nom, 'HR': home_runs})
        except Exception as e:
            st.info(f"Erreur lors de la récupération des Home Runs via statsapi : {e}")
            return []

        result = sorted(result, key=lambda x: x['HR'], reverse=True)
        return result[:top_n]

    # Affichage sous forme de cartes st.metric
    with st.spinner("Recherche des meilleurs frappeurs HR (via MLB StatsAPI)..."):
        top_batteurs_hr = get_top_hr_batteurs_statsapi(annee, equipe_abbr, top_n=3)
        if not top_batteurs_hr:
            st.info("Aucun joueur avec Home Runs enregistré pour cette équipe/saison.")
        else:
            slugger_cols = st.columns(len(top_batteurs_hr))
            for idx, row in enumerate(top_batteurs_hr):
                joueur = row['name']
                hr = row['HR']
                with slugger_cols[idx]:
                    st.metric(label=joueur, value=f"{hr} HR")
    # ----- Fin de la nouvelle logique HR équipe via MLB StatsAPI ------

    st.markdown("---")
    st.subheader("📈 Tendance des Runs par match (score équipe)")
    # 2. Graphique tendance Runs, avec ligne de moyenne annotée
    try:
        if not df_matchs.empty and "R" in df_matchs.columns:
            df_matchs = df_matchs.copy()
            df_matchs['Runs'] = pd.to_numeric(df_matchs['R'], errors='coerce')
            df_matchs = df_matchs.dropna(subset=['Runs'])
            # Ajouter un numéro de match croissant
            df_matchs = df_matchs.reset_index(drop=True)
            df_matchs['Match_Num'] = df_matchs.index + 1

            if not df_matchs.empty:
                moyenne_runs = df_matchs['Runs'].mean()

                ligne_runs = alt.Chart(df_matchs).mark_line(
                    point=True, color='#1f77b4'
                ).encode(
                    x=alt.X('Match_Num:Q', title='Numéro du match'),
                    y=alt.Y('Runs:Q', title='Runs marqués'),
                    tooltip=[
                        alt.Tooltip('Match_Num:Q', title='Match #'),
                        alt.Tooltip('Runs:Q', title='Runs')
                    ]
                )

                ligne_moyenne = alt.Chart(pd.DataFrame({'moyenne': [moyenne_runs]})).mark_rule(
                    color='red', strokeDash=[6, 4], size=2
                ).encode(
                    y=alt.Y('moyenne:Q'),
                    tooltip=[alt.Tooltip('moyenne:Q', title='Moyenne', format='.2f')]
                )

                annotation_moyenne = alt.Chart(pd.DataFrame({
                    'moyenne': [moyenne_runs],
                    'x': [df_matchs['Match_Num'].max()]
                })).mark_text(
                    text=f"Moyenne : {moyenne_runs:.2f}",
                    align='right',
                    baseline='bottom',
                    dx=-4,
                    dy=-6,
                    color='red',
                    fontWeight='bold'
                ).encode(
                    x=alt.X('x:Q'),
                    y=alt.Y('moyenne:Q')
                )

                st.altair_chart(ligne_runs + ligne_moyenne + annotation_moyenne)
            else:
                st.info("Pas de données de runs disponibles pour cette équipe/saison.")
        else:
            st.info("Pas de données de runs disponibles pour cette équipe/saison.")
    except Exception as e:
        st.info(f"Erreur lors de l'affichage des tendances de runs : {e}")

    # Statistiques synthétiques en haut
    if not df_matchs.empty and 'R' in df_matchs.columns:
        runs_total = df_matchs['R'].sum()
        runs_moyen = df_matchs['R'].mean()
        matchs_joues = len(df_matchs[df_matchs['R'].notna()])

        st.markdown(f"### Statistiques des Runs - {EQUIPES_MLB[equipe_abbr]} ({annee})")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric(
                label="Runs Totaux",
                value=f"{int(runs_total)}",
                help="Nombre total de runs marqués cette saison")
        with col_stat2:
            st.metric(
                label="Moyenne par Match",
                value=f"{runs_moyen:.2f}",
                help="Moyenne de runs marqués par match"
            )
        with col_stat3:
            st.metric(
                label="Matchs Analysés",
                value=f"{matchs_joues}",
                help="Nombre de matchs avec données disponibles"
            )

        st.markdown("---")
        st.subheader("📋 Derniers Matchs")
        display_columns = ['Date', 'Équipe Domicile', 'Équipe Extérieur', 'R', 'RA', 'W/L', 'Joueurs (Runs)', 'Joueurs (HR)']
        df_recents = df_matchs.tail(10)
        df_recents = df_recents[display_columns] if all(c in df_recents.columns for c in display_columns) else df_recents

        # Renommer les colonnes pour la présentation
        df_recents = df_recents.rename(columns={
            'R': 'Runs',
            'RA': 'Runs_Adverses',
            'W/L': 'Résultat'
        })

        # --- Ajout du surlignage sur l'équipe sélectionnée dans le tableau des matchs ---

        # Nom de l'équipe sélectionnée (utilisé pour la surbrillance)
        nom_equipe_sel = EQUIPES_MLB.get(equipe_abbr, "")

        def highlight_team(cell):
            if cell == nom_equipe_sel:
                # On utilise un bleu claire qui convient sur clair comme foncé
                return 'background-color: #bdd7ee; font-weight: bold;'
            return ''

        # Affichage du DataFrame stylé
        try:
            st.dataframe(
                df_recents.style.applymap(
                    highlight_team,
                    subset=['Équipe Domicile', 'Équipe Extérieur']
                ),
                use_container_width=True,
                hide_index=True
            )
        except Exception:
            st.dataframe(df_recents, use_container_width=True, hide_index=True)

        # --- Résumé permanent des 10 derniers matchs (se met à jour automatiquement) ---
        moyenne_runs_10, top3_runs_10, moyenne_hr_10, top3_hr_10, cumul_runs_10, cumul_hr_10 = calculer_resume_10_derniers_matchs(
            df_matchs.tail(10)
        )
        if moyenne_runs_10 is not None:
            texte_top3_runs = (
                ", ".join(f"{nom} ({runs} runs)" for nom, runs in top3_runs_10)
                if top3_runs_10 else "Aucun joueur enregistré"
            )
            texte_top3_hr = (
                ", ".join(f"{nom} ({hr} HR)" for nom, hr in top3_hr_10)
                if top3_hr_10 else "Aucun joueur enregistré"
            )

            st.markdown(f"**Moyenne de runs sur les 10 derniers matchs : {moyenne_runs_10:.2f}**")
            st.markdown(f"**Top 3 des joueurs les plus récurrents (runs marqués) : {texte_top3_runs}**")
            st.markdown(f"**Moyenne de home runs sur les 10 derniers matchs : {moyenne_hr_10:.2f}**")
            st.markdown(f"**Top 3 des joueurs les plus récurrents (home runs) : {texte_top3_hr}**")
        else:
            st.markdown("**Résumé indisponible : pas assez de données sur les 10 derniers matchs.**")

        st.markdown("---")
        col_runs, col_hr = st.columns(2)
        with col_runs:
            st.subheader("🏅 Meilleurs scoreurs de Runs")
            st.markdown(f"Cumul des runs marqués par joueur sur la saison {annee}")
            if not df_meilleurs_scoreurs.empty:
                st.dataframe(
                    df_meilleurs_scoreurs,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Aucune donnée de scoreurs disponible pour cette équipe/saison.")
        with col_hr:
            st.subheader("🏆 Meilleurs frappeurs de Home Runs")
            st.markdown(f"Cumul des home runs marqués par joueur sur la saison {annee}")
            if not df_meilleurs_hr.empty:
                st.dataframe(
                    df_meilleurs_hr,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Aucune donnée de home runs disponible pour cette équipe/saison.")
    elif not df_matchs.empty:
        st.warning("Données de runs non disponibles pour cette équipe.")
    else:
        st.error("Impossible de charger les données. Vérifiez l'abréviation de l'équipe.")

# --------------------------------------------------------------
# ONGLET 2: PRÉDICTIONS DU JOUR
# --------------------------------------------------------------
with onglets[1]:
    st.header("🔮 Prédictions du jour")
    st.markdown(f"Prédiction du match du jour pour les **{EQUIPES_MLB.get(equipe_abbr, equipe_abbr)}**")
    st.caption(
        "⚠️ Estimations statistiques basées sur les tendances récentes de l'équipe et les stats du "
        "lanceur adverse. Ce ne sont pas des garanties de résultat : à utiliser uniquement à titre "
        "informatif, avec discernement si vous vous en servez pour parier."
    )

    if annee != ANNEE_COURANTE:
        st.info(
            f"Les prédictions du jour ne sont disponibles que pour la saison en cours "
            f"({ANNEE_COURANTE}). Sélectionnez {ANNEE_COURANTE} dans le menu de gauche pour "
            f"voir la prédiction du match d'aujourd'hui."
        )
    else:
        team_ids_pred = get_team_ids_dict(annee)
        team_id_selectionne = team_ids_pred.get(equipe_abbr)

        maintenant_us_aff = datetime.now(TZ_US_EASTERN)
        maintenant_paris_aff = maintenant_us_aff.astimezone(TZ_PARIS)
        st.caption(
            f"📅 Aujourd'hui aux USA (heure de l'Est) : {maintenant_us_aff.strftime('%A %d %B %Y, %H:%M %Z')} "
            f"— soit {maintenant_paris_aff.strftime('%A %d %B %Y, %H:%M')} en France."
        )

        with st.spinner("Recherche du match du jour..."):
            match_du_jour = obtenir_match_du_jour(team_id_selectionne)

        if not match_du_jour:
            st.info(f"Aucun match n'est prévu aujourd'hui (heure US) pour les {EQUIPES_MLB.get(equipe_abbr, equipe_abbr)}.")
        else:
            lieu = "à domicile" if match_du_jour['est_domicile'] else "à l'extérieur"
            st.subheader(
                f"🆚 {EQUIPES_MLB.get(equipe_abbr, equipe_abbr)} {lieu} contre {match_du_jour['adversaire']}"
            )

            col_venue, col_heure_us, col_heure_paris, col_statut = st.columns(4)
            with col_venue:
                st.metric("Stade", match_du_jour['venue'] or "—")
            with col_heure_us:
                st.metric("Heure (US, Est)", match_du_jour['heure_us'])
            with col_heure_paris:
                st.metric("Heure (France)", match_du_jour['heure_paris'])
            with col_statut:
                st.metric("Statut", match_du_jour['statut'] or "—")

            st.markdown("#### ⚾ Lanceurs partants prévus")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown(f"**{EQUIPES_MLB.get(equipe_abbr, equipe_abbr)}**")
                st.markdown(f"### {match_du_jour['lanceur_notre_equipe'] or 'Non annoncé'}")
            with col_p2:
                st.markdown(f"**{match_du_jour['adversaire']}**")
                st.markdown(f"### {match_du_jour['lanceur_adverse'] or 'Non annoncé'}")

            # Les stats du lanceur adverse ET celles de NOTRE lanceur (saison en cours) sont
            # récupérées de manière SYMÉTRIQUE via `obtenir_stats_lanceur()` - les deux sont
            # nécessaires au module "Probabilité de Victoire" ci-dessous, qui compare les
            # DEUX lanceurs partants prévus (`obtenir_match_du_jour` ne les fournit pas déjà
            # groupées comme dans les apps KBO/NPB, donc on fait les deux appels ici).
            with st.spinner("Analyse des statistiques des lanceurs partants..."):
                stats_lanceur_adverse = obtenir_stats_lanceur(match_du_jour['lanceur_adverse'], annee)
                stats_lanceur_nous = obtenir_stats_lanceur(match_du_jour['lanceur_notre_equipe'], annee)

            if stats_lanceur_adverse:
                st.caption(
                    f"Stats saison {annee} de {stats_lanceur_adverse['nom']} : "
                    f"ERA {stats_lanceur_adverse['era']:.2f} · WHIP {stats_lanceur_adverse['whip']:.2f} · "
                    f"{stats_lanceur_adverse['hr_alloues']} HR alloués · "
                    f"{stats_lanceur_adverse['matchs_titulaire']} départs"
                )
            elif match_du_jour['lanceur_adverse']:
                st.caption("Statistiques du lanceur adverse indisponibles pour le moment.")

            # Moyenne de runs CONCÉDÉS par notre équipe sur ses 10 derniers matchs : calculée
            # UNE SEULE FOIS ici, puis réutilisée à la fois par le module "Probabilité de
            # Victoire" ci-dessous (comme proxy de l'attaque adverse, voir docstring de
            # `predire_probabilite_victoire`) et par le module de prédiction des Runs plus
            # bas (qui l'utilisait déjà comme proxy identique).
            moyenne_ra_10 = pd.to_numeric(
                df_matchs.tail(10).get('RA', pd.Series(dtype=float)), errors='coerce'
            ).mean()

            # --------------------------------------------------------------
            # MODULE : PROBABILITÉ DE VICTOIRE
            # --------------------------------------------------------------
            st.markdown("---")
            st.subheader("🎲 Probabilité de Victoire")

            pct_nous, pct_adverse = predire_probabilite_victoire(
                moyenne_runs_10,
                moyenne_ra_10,
                stats_lanceur_nous,
                stats_lanceur_adverse,
                match_du_jour['est_domicile'],
            )

            col_proba1, col_proba2 = st.columns(2)
            with col_proba1:
                st.metric(f"{EQUIPES_MLB.get(equipe_abbr, equipe_abbr)}", f"{pct_nous:.0f}%")
            with col_proba2:
                st.metric(f"{match_du_jour['adversaire']}", f"{pct_adverse:.0f}%")
            st.progress(pct_nous / 100)

            st.caption(
                "Estimation basée sur (1) l'ERA/WHIP des lanceurs partants prévus des deux "
                "équipes (facteur principal), (2) la moyenne de runs marqués/concédés sur les "
                "10 derniers matchs (dynamique offensive récente), et (3) un léger bonus de "
                "+3 points de pourcentage pour l'équipe qui joue à domicile (~53-54% de "
                "victoires à domicile en moyenne dans le baseball professionnel). "
                "⚠️ Simple heuristique, PAS un modèle statistique validé : ne reflète pas "
                "tous les facteurs d'un vrai match (composition exacte de l'équipe, bullpen, "
                "météo, blessures de dernière minute, etc.)."
            )

            lanceur_nous_ok = bool(stats_lanceur_nous and stats_lanceur_nous.get('era'))
            lanceur_adv_ok = bool(stats_lanceur_adverse and stats_lanceur_adverse.get('era'))
            if not (lanceur_nous_ok and lanceur_adv_ok):
                st.info(
                    "ℹ️ Stats ERA/WHIP indisponibles pour au moins un des deux lanceurs "
                    "prévus (facteur neutralisé pour le(s) lanceur(s) concerné(s)) : "
                    "l'estimation ci-dessus est donc moins fiable que d'habitude."
                )

            st.markdown("---")
            st.subheader("📊 Module de prédiction des Runs")

            if moyenne_runs_10 is None:
                st.info("Pas assez de données récentes pour estimer les runs de cette équipe.")
            else:
                prediction_runs = predire_runs_match(moyenne_runs_10, moyenne_ra_10, stats_lanceur_adverse)

                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric(
                        f"Runs estimés — {equipe_abbr}",
                        f"{prediction_runs['runs_equipe']}"
                    )
                with col_pred2:
                    st.metric("Total de runs estimé (match)", f"{prediction_runs['total_match']}")
                with col_pred3:
                    st.metric("Indice de confiance", prediction_runs['confiance'])

                st.caption(
                    f"Basé sur une moyenne de {moyenne_runs_10:.2f} runs/match et "
                    f"{moyenne_ra_10:.2f} runs concédés/match sur les 10 derniers matchs, "
                    + (
                        f"croisée avec les stats du lanceur adverse ({stats_lanceur_adverse['nom']})."
                        if stats_lanceur_adverse
                        else "en l'absence de stats fiables sur le lanceur adverse."
                    )
                )

            st.markdown("---")
            st.subheader("🎯 Module de prédiction des Joueurs (HR / Runs)")

            joueurs_a_surveiller = predire_joueurs_du_jour(
                cumul_runs_10, cumul_hr_10, stats_lanceur_adverse, top_n=3
            )

            if not joueurs_a_surveiller:
                st.info(
                    "Pas assez de données de forme récente (runs/HR sur les 10 derniers matchs) "
                    "pour identifier des joueurs à surveiller aujourd'hui."
                )
            else:
                cols_joueurs = st.columns(len(joueurs_a_surveiller))
                for idx, joueur in enumerate(joueurs_a_surveiller):
                    with cols_joueurs[idx]:
                        st.markdown(f"**{joueur['nom']}**")
                        st.progress(joueur['indice'] / 100)
                        st.markdown(f"Indice de confiance : **{joueur['confiance']}** ({joueur['indice']}/100)")
                        st.caption(
                            f"{joueur['runs_10']} run(s) et {joueur['hr_10']} HR sur les 10 derniers matchs"
                        )

# ============================================================
# 7. PIED DE PAGE
# ============================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>"
    f"⚾ Application MLB Analytics | Données mises à jour: {datetime.now().strftime('%Y-%m-%d')}"
    f"</div>",
    unsafe_allow_html=True
)