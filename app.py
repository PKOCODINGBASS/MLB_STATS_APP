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
import json                     # Sérialisation de l'historique des prédictions (bilan de la veille)
import os                       # Chemin du fichier local d'historique des prédictions
import requests                 # Appels directs à l'API GitHub Gist et à The-Odds-API (Value Bet)
import unicodedata              # Normalisation des noms d'équipe (Value Bet Detector)
from datetime import datetime, timedelta  # Gestion des dates (timedelta : calcul de "hier")
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

# ------------------------------------------------------------------------------
# Persistance de l'historique des prédictions (pour le "Bilan des Prédictions" de la
# veille, onglet Résumé) : un instantané des prédictions du jour ("Hot Pronostics")
# est archivé chaque jour, pour pouvoir être comparé au résultat réel le lendemain.
#
# Streamlit Community Cloud utilise un système de fichiers ÉPHÉMÈRE : tout fichier
# écrit localement pendant l'exécution est PERDU à chaque redéploiement (déclenché par
# un `git push`) ou "réveil" de l'app après une période d'inactivité. Un simple fichier
# local ne suffit donc pas à conserver l'historique dans la durée sur cet hébergement.
#
# La source de vérité est donc un Gist GitHub PRIVÉ (persiste indéfiniment, quel que
# soit le nombre de redéploiements), configuré via `st.secrets` :
#
#     [github]
#     token = "ghp_..."   # Personal Access Token GitHub, scope "gist" UNIQUEMENT
#     gist_id = "..."     # ID du Gist privé contenant historique_predictions_mlb.json
#
# à renseigner dans `.streamlit/secrets.toml` en local, et dans les "Secrets" de l'app
# sur share.streamlit.io en production (jamais commités : `.streamlit/secrets.toml`
# est listé dans `.gitignore`).
#
# Si ces secrets ne sont pas configurés (ex: tout premier lancement, développement
# local sans Gist créé), l'application se rabat silencieusement sur le fichier local
# ci-dessous - fonctionnel, mais non persistant sur Streamlit Cloud. Ce fichier local
# sert aussi de cache accessoire même quand le Gist est configuré (repli en cas de
# panne réseau GitHub ponctuelle).
# ------------------------------------------------------------------------------
NOM_FICHIER_HISTORIQUE_PREDICTIONS = "historique_predictions_mlb.json"
CHEMIN_HISTORIQUE_PREDICTIONS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), NOM_FICHIER_HISTORIQUE_PREDICTIONS
)


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


def _obtenir_config_github():
    """
    Lit la configuration GitHub (token + ID du Gist privé) dans `st.secrets`, utilisée
    pour la persistance durable de l'historique des prédictions (cf. commentaire au-
    dessus de `CHEMIN_HISTORIQUE_PREDICTIONS`). Retourne (token, gist_id), ou
    (None, None) si non configuré - jamais d'exception : accéder à `st.secrets` lève
    une erreur s'il n'existe AUCUN fichier `secrets.toml` du tout (cas du tout premier
    lancement / développement local sans Gist configuré), qu'il faut absorber ici pour
    retomber sur le fichier local en toute transparence.
    """
    try:
        conf = st.secrets.get("github", {})
        return conf.get("token"), conf.get("gist_id")
    except Exception:
        return None, None


def _charger_historique_predictions() -> dict:
    """
    Charge l'historique des prédictions archivées (un instantané par date, au format
    {'AAAA-MM-JJ': {'sauvegarde_le': ..., 'matches': [...]}}) - en PRIORITÉ depuis le
    Gist GitHub privé configuré (`_obtenir_config_github`), seule source qui survit aux
    redéploiements sur Streamlit Community Cloud. Repli sur le fichier local
    `CHEMIN_HISTORIQUE_PREDICTIONS` si le Gist n'est pas configuré, ou si l'appel à
    l'API GitHub échoue (panne réseau ponctuelle, token invalide, etc.).

    Retourne un dict vide si aucune des deux sources n'est disponible (ex: tout premier
    lancement de l'application) - ne doit jamais faire planter l'application.
    """
    token, gist_id = _obtenir_config_github()
    if token and gist_id:
        try:
            reponse = requests.get(
                f"https://api.github.com/gists/{gist_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
            reponse.raise_for_status()
            fichier = reponse.json().get("files", {}).get(NOM_FICHIER_HISTORIQUE_PREDICTIONS)
            if fichier and fichier.get("content"):
                return json.loads(fichier["content"])
            return {}
        except Exception:
            pass  # repli silencieux sur le fichier local ci-dessous

    try:
        with open(CHEMIN_HISTORIQUE_PREDICTIONS, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _sauvegarder_predictions_du_jour(date_str: str, matches_snapshot: list) -> None:
    """
    Archive l'instantané des prédictions du jour (`matches_snapshot`) sous la clé
    `date_str`, à la fois dans le Gist GitHub privé configuré (source durable, cf.
    `_obtenir_config_github`) ET dans le fichier local (repli/cache accessoire).
    Appelée depuis `construire_donnees_hot_pronostics` (donc au maximum une fois
    toutes les 30 min, son propre `ttl` de cache) : écrire à chaque appel écrase
    simplement l'instantané du jour par la version la plus à jour (utile si les
    lineups/lanceurs annoncés changent en cours de journée), ce qui est le
    comportement recherché.

    Purge au passage les entrées de plus de 30 jours, pour que l'historique ne
    grossisse pas indéfiniment au fil des mois. Ne lève jamais d'exception : la
    sauvegarde de l'historique est un "bonus" (bilan de la veille) qui ne doit jamais
    faire planter le calcul des prédictions du jour lui-même en cas de souci réseau ou
    d'écriture disque (permissions, disque plein, filesystem éphémère, etc.).
    """
    try:
        historique = _charger_historique_predictions()
        historique[date_str] = {
            'sauvegarde_le': datetime.now(TZ_US_EASTERN).isoformat(),
            'matches': matches_snapshot,
        }
        date_limite = (datetime.now(TZ_US_EASTERN) - timedelta(days=30)).strftime('%Y-%m-%d')
        historique = {d: v for d, v in historique.items() if d >= date_limite}
        contenu_json = json.dumps(historique, ensure_ascii=False, indent=2)

        token, gist_id = _obtenir_config_github()
        if token and gist_id:
            try:
                reponse = requests.patch(
                    f"https://api.github.com/gists/{gist_id}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                    },
                    json={"files": {NOM_FICHIER_HISTORIQUE_PREDICTIONS: {"content": contenu_json}}},
                    timeout=10,
                )
                reponse.raise_for_status()
            except Exception:
                pass  # au pire, le fichier local ci-dessous prend seul le relais

        with open(CHEMIN_HISTORIQUE_PREDICTIONS, "w", encoding="utf-8") as f:
            f.write(contenu_json)
    except Exception:
        pass


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


# --------------------------------------------------------------
# SEUILS DE PRÉDICTION PAR LIGUE ("Recommandation de Pari Optimisée")
# --------------------------------------------------------------
# Les moyennes offensives et l'ERA "normal" diffèrent fortement d'une ligue à l'autre
# (ex: la MLB est réputée plus équilibrée/offensive que la NPB ou la KBO). Centraliser
# ces seuils dans un dictionnaire clé = code ligue permet à `generer_recommandation_pari`
# de s'adapter automatiquement à la ligue du match en cours (voir `detecter_ligue_match`),
# sans jamais coder les seuils MLB "en dur" dans la logique elle-même - une future
# extension multi-ligues (ex: import des apps NPB/KBO dans ce même projet) n'aurait
# qu'à ajouter une entrée ici.
LIGUE_PAR_DEFAUT = 'MLB'

SEUILS_PARIS_PAR_LIGUE = {
    'MLB': {
        # ERA d'un lanceur partant jugé "battable" (favorise un pari Over)
        'era_mauvais': 4.50,
        # ERA d'un lanceur partant jugé dominant (favorise un pari Under)
        'era_excellent': 3.50,
        # Total de runs (des deux équipes cumulé) au-delà duquel on considère la
        # tendance du match comme offensive (MLB = ligue équilibrée/offensive,
        # seuil plus haut qu'en NPB/KBO par exemple)
        'runs_total_haut': 9.0,
    },
}


def detecter_ligue_match(match_du_jour: dict = None) -> str:
    """
    Détecte la ligue du match en cours à partir des infos du match (`obtenir_match_du_jour`),
    afin que `generer_recommandation_pari` applique les bons seuils ERA/Runs (voir
    `SEUILS_PARIS_PAR_LIGUE`). Cette application ne couvre aujourd'hui que la MLB
    (source de données unique : MLB StatsAPI), donc le résultat vaut toujours 'MLB' en
    pratique - mais la détection passe bien par le champ `ligue` du match (plutôt
    qu'un `LIGUE_PAR_DEFAUT` codé en dur dans l'appelant), pour que la logique reste
    correcte sans modification si l'app venait à couvrir plusieurs ligues (ex: KBO, NPB).
    """
    if match_du_jour and match_du_jour.get('ligue'):
        return match_du_jour['ligue']
    return LIGUE_PAR_DEFAUT


def generer_recommandation_pari(
    pct_nous,
    pct_adverse,
    stats_lanceur_nous,
    stats_lanceur_adverse,
    prediction_runs,
    joueurs_a_surveiller,
    ligue: str = None,
    vent_defavorable: bool = False,
):
    """
    Génère la "Recommandation de Pari Optimisée" affichée sous la ligne principale de
    prédiction (probabilité de victoire) de l'onglet "Prédictions du jour", via un petit
    arbre de décision qui croise plusieurs facteurs déjà calculés ailleurs dans l'onglet.
    Objectif affiché à l'utilisateur : minimiser le risque, pas maximiser le gain.

    --- Étape 1 : Risque sur le résultat (Win/Loss) - universel, toutes ligues ---
    Évalue systématiquement la "qualité" du match du point de vue du pari vainqueur
    (une phrase est TOUJOURS générée à cette étape, contrairement aux étapes 2 et 3) :
      - Si l'écart entre les deux probabilités de victoire est inférieur à 10 points, le
        match est jugé "à Haut Risque" sur le vainqueur : on recommande de préférer un
        pari sur les runs plutôt que sur le résultat (moins dépendant d'un seul évènement).
      - Sinon (écart >= 10 points, un favori se dégage nettement), le match est jugé
        "à Faible Risque" sur le vainqueur : un pari sur le résultat est alors présenté
        comme une option plus fiable qu'un pari sur les runs.

    --- Étape 2 : Total de runs (Over/Under) - seuils spécifiques à la ligue ---
    Seuils lus dans `SEUILS_PARIS_PAR_LIGUE[ligue]` (repli sur `LIGUE_PAR_DEFAUT` si la
    ligue est inconnue) :
      - Condition "tendance haute" (Over) : les DEUX lanceurs partants prévus ont un
        ERA supérieur au seuil "mauvais ERA" de la ligue, OU le total de runs estimé du
        match dépasse le seuil "runs haut" de la ligue.
      - Condition "tendance basse" (Under) : les DEUX lanceurs ont un ERA inférieur au
        seuil "excellent ERA" de la ligue, OU le vent est défavorable aux frappeurs
        (facteur météo optionnel, non disponible aujourd'hui côté MLB StatsAPI - prévu
        pour une future intégration, `vent_defavorable=False` par défaut).
      La ligne de total proposée est décalée de 1.5 run (arrondi au 0,5 le plus proche)
      DANS LE SENS QUI RÉDUIT LE RISQUE : en dessous de l'estimation pour un Over, au-dessus
      pour un Under, pour se laisser une marge plutôt que de parier pile sur l'estimation brute.
      Une phrase Over/Under est TOUJOURS générée dès que le total estimé est disponible
      (repli : Over si projection >= seuil haut de ligue, sinon Under), y compris quand
      l'étape 1 privilégie déjà un pari sur le vainqueur.

    --- Étape 3 : Option joueur (HR/Run) - universel ---
    Si un joueur du module "Prédiction des Joueurs" (nos sluggers en forme du jour,
    `joueurs_a_surveiller`) ressort avec une confiance au moins "Moyenne", il est proposé
    comme option alternative de pari.

    Retourne une liste de phrases (str), dans l'ordre ci-dessus, prête à être jointe et
    affichée dans un seul encart (ex: `st.info`). Liste vide si aucune recommandation
    n'a pu être formulée (données insuffisantes).
    """
    ligue = ligue or LIGUE_PAR_DEFAUT
    seuils = SEUILS_PARIS_PAR_LIGUE.get(ligue, SEUILS_PARIS_PAR_LIGUE[LIGUE_PAR_DEFAUT])

    def _arrondir_au_demi(valeur: float) -> float:
        """Arrondit au 0,5 le plus proche (ex: 8.2 -> 8.0, 8.3 -> 8.5)."""
        return round(valeur * 2) / 2

    def _era(stats):
        return stats['era'] if stats and stats.get('era') else None

    conseils = []

    # --- Étape 1 : risque Win/Loss (universel) - toujours une phrase, dans un sens ou l'autre ---
    if pct_nous is not None and pct_adverse is not None:
        if abs(pct_nous - pct_adverse) < 10:
            conseils.append(
                "⚠️ Match serré (Haut Risque sur la victoire). Privilégiez un pari sur "
                "les Runs plutôt que sur le vainqueur."
            )
        else:
            favori = "notre équipe" if pct_nous > pct_adverse else "l'équipe adverse"
            conseils.append(
                f"✅ Écart de probabilité net en faveur de {favori} (Faible Risque sur la "
                "victoire). Un pari sur le vainqueur est ici plus fiable qu'un pari sur les Runs."
            )

    # --- Étape 2 : total de runs Over/Under (spécifique à la ligue) ---
    # Toujours une phrase Over/Under dès que le total estimé est disponible, y compris
    # quand l'étape 1 privilégie déjà un pari sur le vainqueur (favori net) : le conseil
    # runs reste alors une option complémentaire utile.
    era_nous = _era(stats_lanceur_nous)
    era_adverse = _era(stats_lanceur_adverse)
    deux_lanceurs_connus = era_nous is not None and era_adverse is not None

    deux_mauvais_era = deux_lanceurs_connus and era_nous > seuils['era_mauvais'] and era_adverse > seuils['era_mauvais']
    deux_excellents_era = deux_lanceurs_connus and era_nous < seuils['era_excellent'] and era_adverse < seuils['era_excellent']

    total_runs_estime = prediction_runs.get('total_match') if prediction_runs else None
    tendance_offensive_runs = total_runs_estime is not None and total_runs_estime > seuils['runs_total_haut']

    if total_runs_estime is not None:
        if deux_mauvais_era or tendance_offensive_runs:
            ligne_over = _arrondir_au_demi(total_runs_estime - 1.5)
            conseils.append(
                f"📈 Tendance offensive forte. Conseil : Jouer 'Over {ligne_over} runs'."
            )
        elif deux_excellents_era or vent_defavorable:
            ligne_under = _arrondir_au_demi(total_runs_estime + 1.5)
            conseils.append(
                f"📉 Match très défensif anticipé. Conseil : Jouer 'Under {ligne_under} runs'."
            )
        elif total_runs_estime >= seuils['runs_total_haut']:
            ligne_over = _arrondir_au_demi(total_runs_estime - 1.5)
            conseils.append(
                f"📈 Projection de runs au seuil haut de la ligue. Conseil : Jouer "
                f"'Over {ligne_over} runs'."
            )
        else:
            ligne_under = _arrondir_au_demi(total_runs_estime + 1.5)
            conseils.append(
                f"📉 Projection de runs contenue. Conseil : Jouer 'Under {ligne_under} runs'."
            )

    # --- Étape 3 : option joueur (universel) ---
    if joueurs_a_surveiller:
        meilleur_joueur = joueurs_a_surveiller[0]
        if meilleur_joueur.get('confiance') in ('Élevée', 'Moyenne'):
            conseils.append(
                f"🎯 Option alternative : {meilleur_joueur['nom']} a une forte probabilité "
                "de marquer un Run/HR aujourd'hui."
            )

    return conseils


# --------------------------------------------------------------
# VALUE BET DETECTOR (comparaison avec les cotes Winamax / marché)
# --------------------------------------------------------------
# Source de cotes : The-Odds-API (https://the-odds-api.com), qui agrège de nombreux
# bookmakers dont Winamax (clé bookmaker 'winamax_fr', région 'eu') - Winamax n'ayant
# pas d'API publique/officielle, passer par cet agrégateur évite le scraping direct de
# leur site (fragile et probablement contraire à leurs CGU) tout en donnant accès à
# leurs cotes réelles quand ce bookmaker couvre le match.
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'
ODDS_API_SPORT_KEY = 'baseball_mlb'
ODDS_API_BOOKMAKER_PRINCIPAL = 'winamax_fr'
# Région de repli si Winamax ne propose pas (encore) de cote sur ce match précis -
# on retombe alors sur le 1er bookmaker EU disponible plutôt que d'afficher
# "indisponible" alors qu'une cote de marché existe ailleurs.
ODDS_API_REGION = 'eu'


def _lire_cle_odds_api():
    """
    Lit la clé API The-Odds-API dans `st.secrets` (section [odds_api], clé `api_key`),
    utilisée par le "Value Bet Detector". Retourne None si non configurée - jamais
    d'exception : accéder à `st.secrets` lève une erreur si le fichier secrets.toml
    n'existe pas du tout, d'où le `try/except` (même pattern que la config GitHub
    utilisée pour la persistance de l'historique des prédictions).
    """
    try:
        conf = st.secrets.get("odds_api", {})
        return conf.get("api_key")
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def obtenir_cotes_moneyline_du_jour(sport_key: str, api_key: str):
    """
    Récupère, via The-Odds-API, les cotes "Moneyline" (marché h2h = vainqueur du
    match, sans handicap) de TOUS les matchs à venir aujourd'hui pour le sport/ligue
    demandé (`sport_key`, ex: 'baseball_mlb'), en priorité chez Winamax
    (`ODDS_API_BOOKMAKER_PRINCIPAL`). Si Winamax ne propose pas ce marché pour un
    match donné, on retombe sur le 1er bookmaker EU disponible pour ce match plutôt
    que de le considérer comme "indisponible" alors qu'une cote de marché existe.

    Mise en cache 30 minutes : le quota gratuit de The-Odds-API est limité (500
    requêtes/mois), inutile de rappeler l'API à chaque interaction utilisateur pour
    des cotes qui ne bougent pas d'une minute à l'autre.

    Retourne une liste de dicts {'equipe_domicile', 'equipe_exterieur',
    'cote_domicile', 'cote_exterieur', 'bookmaker'} (une entrée par match), ou []
    si la clé API n'est pas configurée, si la ligue n'est pas couverte aujourd'hui,
    ou en cas d'erreur réseau/API (ex: quota dépassé) - jamais d'exception remontée
    à l'appelant.
    """
    if not api_key or not sport_key:
        return []
    try:
        reponse = requests.get(
            f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds",
            params={
                'apiKey': api_key,
                'regions': ODDS_API_REGION,
                'markets': 'h2h',
                'oddsFormat': 'decimal',
            },
            timeout=10,
        )
        reponse.raise_for_status()
        matchs_api = reponse.json()
    except Exception:
        return []

    resultats = []
    for match in matchs_api:
        bookmakers = match.get('bookmakers') or []
        bookmaker_retenu = next(
            (b for b in bookmakers if b.get('key') == ODDS_API_BOOKMAKER_PRINCIPAL),
            bookmakers[0] if bookmakers else None,
        )
        if not bookmaker_retenu:
            continue
        marche_h2h = next(
            (m for m in bookmaker_retenu.get('markets', []) if m.get('key') == 'h2h'), None
        )
        if not marche_h2h or len(marche_h2h.get('outcomes', [])) < 2:
            continue
        cotes_par_equipe = {o.get('name'): o.get('price') for o in marche_h2h['outcomes']}
        resultats.append({
            'equipe_domicile': match.get('home_team'),
            'equipe_exterieur': match.get('away_team'),
            'cote_domicile': cotes_par_equipe.get(match.get('home_team')),
            'cote_exterieur': cotes_par_equipe.get(match.get('away_team')),
            'bookmaker': bookmaker_retenu.get('title') or bookmaker_retenu.get('key'),
        })
    return resultats


def _normaliser_nom_equipe(texte: str) -> str:
    """Normalise un nom d'équipe (minuscules, sans accents) pour une comparaison assouplie."""
    return unicodedata.normalize('NFKD', texte or '').encode('ascii', 'ignore').decode().lower().strip()


def trouver_cote_du_match(cotes_du_jour: list, nom_notre_equipe: str, nom_adversaire: str):
    """
    Retrouve, dans la liste retournée par `obtenir_cotes_moneyline_du_jour`, le match
    correspondant à notre équipe/adversaire du jour, et renvoie la cote de CHAQUE
    équipe ainsi que le bookmaker utilisé. La correspondance se fait par comparaison
    "assouplie" (sous-chaîne, insensible à la casse/accents) plutôt qu'une égalité
    stricte : les noms d'équipe fournis par The-Odds-API ne correspondent pas toujours
    mot pour mot aux noms utilisés ailleurs dans l'app.

    Retourne un dict {'cote_nous', 'cote_adverse', 'bookmaker'}, ou None si aucun
    match correspondant n'a été trouvé (ligue/bookmaker ne couvrant pas ce match, ou
    marché pas encore ouvert aux paris).
    """
    nous = _normaliser_nom_equipe(nom_notre_equipe)
    adverse = _normaliser_nom_equipe(nom_adversaire)
    if not nous or not adverse:
        return None

    def _correspond(a, b):
        return bool(a) and bool(b) and (a in b or b in a)

    for match in cotes_du_jour:
        dom = _normaliser_nom_equipe(match.get('equipe_domicile'))
        ext = _normaliser_nom_equipe(match.get('equipe_exterieur'))

        if _correspond(nous, dom) and _correspond(adverse, ext):
            return {
                'cote_nous': match.get('cote_domicile'),
                'cote_adverse': match.get('cote_exterieur'),
                'bookmaker': match.get('bookmaker'),
            }
        if _correspond(nous, ext) and _correspond(adverse, dom):
            return {
                'cote_nous': match.get('cote_exterieur'),
                'cote_adverse': match.get('cote_domicile'),
                'bookmaker': match.get('bookmaker'),
            }
    return None


def evaluer_value_bet(proba_algo_pct, cote, nom_equipe: str, nom_bookmaker: str = "Winamax"):
    """
    Compare notre probabilité de victoire estimée (`proba_algo_pct`, calculée par
    `predire_probabilite_victoire`) à la probabilité IMPLICITE de la cote de marché
    (`cote`, au format décimal), pour détecter une éventuelle "Value Bet".

    Probabilité implicite = (1 / cote) * 100.
    Value = Proba_Algo - Proba_Implicite.

    Seuils (identiques pour toutes les ligues - écart de probabilité brut, indépendant
    du profil offensif de la ligue) :
      - Value >= +5 points : le marché sous-évalue cette équipe (badge vert 🟢).
      - Value <= -5 points : le marché la sur-évalue par rapport à notre modèle,
        mieux vaut éviter un pari vainqueur sur cette équipe (badge rouge 🔴).
      - Entre les deux : cote jugée "juste" (badge gris ⚪), pas d'avantage
        mathématique net dans un sens ou l'autre.

    --- IMPORTANT : `nom_bookmaker` ---
    Winamax ne couvre PAS tous les matchs de toutes les ligues (constaté : 0% de
    couverture NPB/KBO chez The-Odds-API contre 100% en MLB). `trouver_cote_du_match`
    retombe alors sur un autre bookmaker EU disponible (voir `ODDS_API_BOOKMAKER_PRINCIPAL`)
    - le message doit donc TOUJOURS citer le bookmaker RÉELLEMENT utilisé (`cotes_match
    ['bookmaker']` côté appelant), jamais "Winamax" en dur, pour ne jamais afficher une
    fausse attribution.

    Retourne un tuple (niveau, message) où niveau vaut 'value', 'juste' ou 'evitez',
    ou (None, None) si la cote n'est pas exploitable (absente ou <= 1.0) ou si la
    probabilité de l'algo est inconnue.
    """
    if not cote or cote <= 1.0 or proba_algo_pct is None:
        return None, None

    proba_implicite = (1.0 / cote) * 100.0
    value = proba_algo_pct - proba_implicite

    if value >= 5:
        return 'value', (
            f"🟢 🔥 Value Bet détectée ! {nom_bookmaker} sous-évalue {nom_equipe} "
            f"(Cote : {cote:.2f}, Value : +{value:.1f}%)."
        )
    if value <= -5:
        return 'evitez', (
            f"🔴 ⛔ Ne pas jouer la Win sur {nom_equipe}. La cote de {nom_bookmaker} "
            f"({cote:.2f}) est trop basse par rapport à nos estimations (Value : {value:.1f}%)."
        )
    return 'juste', (
        f"⚪ ⚖️ Cote juste (Fair Value) sur {nom_equipe} (Cote : {cote:.2f}, {nom_bookmaker}). "
        "Pas d'avantage mathématique majeur."
    )


# ============================================================
# 3 bis. "HOT PRONOSTICS" - Scan GLOBAL de tous les matchs du jour
# ============================================================
# Contrairement aux fonctions ci-dessus (centrées sur UNE équipe sélectionnée dans la
# sidebar), ce bloc analyse TOUS les matchs prévus aujourd'hui (heure US), tous équipes
# confondues, pour en extraire les meilleurs pronostics HR / Runs / Victoire du jour.
# Chaque fonction ci-dessous est mise en cache (@st.cache_data) car ce sont des calculs
# globaux coûteux (plusieurs dizaines de joueurs/lanceurs) qui ne doivent PAS être
# relancés à chaque interaction utilisateur (changement d'équipe/saison dans la
# sidebar) - seulement rafraîchis périodiquement (ttl) pour suivre les lineups qui se
# précisent au fil de la journée.

def _parser_stat_flottant(valeur) -> float:
    """
    Convertit une valeur de statistique statsapi en float, en gérant les cas où l'API
    renvoie une chaîne non numérique (ex: '.---' pour un joueur sans at-bat) plutôt
    qu'un nombre ou une chaîne vide.
    """
    try:
        return float(valeur)
    except (TypeError, ValueError):
        return 0.0


@st.cache_data(show_spinner=False, ttl=1800)
def obtenir_calendrier_jour_avec_lineups(annee: int):
    """
    Récupère TOUS les matchs prévus aujourd'hui (date du jour AUX USA, heure de l'Est -
    voir justification dans `obtenir_match_du_jour`) via UN SEUL appel à l'endpoint
    `schedule` hydraté avec `lineups,probablePitcher`, qui fournit en une fois : les
    deux lanceurs partants prévus ET, quand elles sont déjà publiées par les équipes
    (généralement 1 à 3h avant le match), les compositions d'équipe (lineup) dans
    l'ORDRE de passage au bâton (position 1 = 1er de la liste, etc.).

    Retourne une liste de dicts (un par match), avec une lineup vide ([]) pour un
    camp si elle n'est pas encore annoncée - les tableaux "Hot Pronostics" qui se
    basent sur les lineups s'adaptent en conséquence (ce match ne contribue tout
    simplement pas encore de candidats HR/Run pour ce camp).
    """
    if annee != ANNEE_COURANTE:
        return []

    aujourdhui_us = datetime.now(TZ_US_EASTERN).strftime('%Y-%m-%d')
    try:
        reponse = appeler_avec_retry(
            statsapi.get, 'schedule',
            {'date': aujourdhui_us, 'sportId': 1, 'hydrate': 'lineups,probablePitcher'}
        )
    except Exception:
        return []

    matchs = []
    for bloc_date in reponse.get('dates', []):
        for g in bloc_date.get('games', []):
            equipes = g.get('teams', {}) or {}
            home = equipes.get('home', {}) or {}
            away = equipes.get('away', {}) or {}
            home_team = home.get('team', {}) or {}
            away_team = away.get('team', {}) or {}
            home_pitcher = home.get('probablePitcher') or {}
            away_pitcher = away.get('probablePitcher') or {}

            lineups = g.get('lineups', {}) or {}
            home_lineup = [
                {'id': j.get('id'), 'nom': j.get('fullName'), 'position': idx + 1}
                for idx, j in enumerate(lineups.get('homePlayers', []) or [])
                if j.get('id')
            ]
            away_lineup = [
                {'id': j.get('id'), 'nom': j.get('fullName'), 'position': idx + 1}
                for idx, j in enumerate(lineups.get('awayPlayers', []) or [])
                if j.get('id')
            ]

            # Double fuseau horaire (même logique que `obtenir_match_du_jour`)
            heure_us_str, heure_paris_str = None, None
            game_datetime_str = g.get('gameDate')
            if game_datetime_str:
                try:
                    dt_utc = datetime.fromisoformat(game_datetime_str.replace('Z', '+00:00'))
                    heure_us_str = dt_utc.astimezone(TZ_US_EASTERN).strftime('%H:%M %Z')
                    heure_paris_str = dt_utc.astimezone(TZ_PARIS).strftime('%d/%m à %H:%M')
                except Exception:
                    pass

            matchs.append({
                'game_id': g.get('gamePk'),
                'home_id': home_team.get('id'),
                'home_name': home_team.get('name'),
                'away_id': away_team.get('id'),
                'away_name': away_team.get('name'),
                'home_pitcher_id': home_pitcher.get('id'),
                'home_pitcher_name': home_pitcher.get('fullName'),
                'away_pitcher_id': away_pitcher.get('id'),
                'away_pitcher_name': away_pitcher.get('fullName'),
                'home_lineup': home_lineup,
                'away_lineup': away_lineup,
                'venue': (g.get('venue') or {}).get('name'),
                'statut': ((g.get('status') or {}).get('detailedState')),
                'heure_us': heure_us_str or "—",
                'heure_paris': heure_paris_str or "—",
            })
    return matchs


@st.cache_data(show_spinner=False, ttl=1800)
def obtenir_moyennes_runs_recentes_toutes_equipes(annee: int, jours_historique: int = 25):
    """
    Calcule, pour CHAQUE équipe MLB, la moyenne de runs marqués sur ses 10 derniers
    matchs terminés (Final) - via UN SEUL appel `schedule` couvrant une fenêtre de
    date récente (tous équipes confondues), plutôt qu'un appel par équipe : l'API ne
    permet pas de demander directement "10 derniers matchs, toutes équipes" en un
    seul filtre, mais parcourir un large historique commun revient bien moins cher
    que 30 appels individuels.
    `jours_historique=25` donne une marge confortable au-dessus de 10 matchs même en
    tenant compte des jours de repos/reports (une équipe MLB joue ~5-6 matchs/semaine).
    """
    if annee != ANNEE_COURANTE:
        return {}

    aujourdhui = datetime.now(TZ_US_EASTERN)
    date_debut = (aujourdhui - pd.Timedelta(days=jours_historique)).strftime('%Y-%m-%d')
    date_fin = aujourdhui.strftime('%Y-%m-%d')
    try:
        matchs = appeler_avec_retry(
            statsapi.schedule, start_date=date_debut, end_date=date_fin, sportId=1
        )
    except Exception:
        return {}

    matchs_par_equipe = {}
    for g in matchs:
        if g.get('status') != 'Final':
            continue
        home_id, away_id = g.get('home_id'), g.get('away_id')
        home_score, away_score = g.get('home_score'), g.get('away_score')
        if home_score is None or away_score is None:
            continue
        date_match = g.get('game_date', '')
        if home_id:
            matchs_par_equipe.setdefault(home_id, []).append((date_match, home_score))
        if away_id:
            matchs_par_equipe.setdefault(away_id, []).append((date_match, away_score))

    moyennes = {}
    for team_id, liste in matchs_par_equipe.items():
        dix_derniers = sorted(liste, key=lambda x: x[0])[-10:]
        if dix_derniers:
            moyennes[team_id] = sum(r for _, r in dix_derniers) / len(dix_derniers)
    return moyennes


@st.cache_data(show_spinner=False, ttl=1800)
def obtenir_stats_lanceurs_par_id(pitcher_ids: tuple):
    """
    Récupère ERA / WHIP / HR-par-9 (saison en cours) pour un ENSEMBLE de lanceurs en
    un minimum d'appels API : l'endpoint `people` accepte une liste de `personIds`
    séparés par des virgules et une hydratation de stats groupée, ce qui permet de
    couvrir tous les lanceurs partants prévus aujourd'hui (jusqu'à ~30) en 1 seul
    appel au lieu d'un appel par lanceur (comme le ferait `obtenir_stats_lanceur`,
    conservée telle quelle pour l'onglet "Prédictions du jour" mono-équipe).
    """
    ids = sorted({str(i) for i in pitcher_ids if i})
    resultats = {}
    if not ids:
        return resultats

    for debut in range(0, len(ids), 40):  # lots de 40 IDs pour rester sur des URLs raisonnables
        lot = ids[debut:debut + 40]
        try:
            reponse = appeler_avec_retry(
                statsapi.get, 'people',
                {'personIds': ','.join(lot), 'hydrate': 'stats(group=[pitching],type=[season])'}
            )
        except Exception:
            continue
        for p in reponse.get('people', []):
            for bloc in p.get('stats', []):
                if bloc.get('type', {}).get('displayName') != 'season':
                    continue
                splits = bloc.get('splits') or []
                if not splits:
                    continue
                s = splits[0].get('stat', {})
                if not s.get('era'):
                    continue
                resultats[p['id']] = {
                    'nom': p.get('fullName'),
                    'era': _parser_stat_flottant(s.get('era')),
                    'whip': _parser_stat_flottant(s.get('whip')),
                    'hr_par_9': _parser_stat_flottant(s.get('homeRunsPer9')),
                }
    return resultats


@st.cache_data(show_spinner=False, ttl=1800)
def obtenir_stats_batteurs_par_id(batter_ids: tuple):
    """
    Récupère, pour un ENSEMBLE de batteurs (lineups du jour), leur SLG/OBP de saison
    ET sur leurs 10 derniers matchs (`type=lastXGames,limit=10` - stat officielle
    statsapi équivalente à une "forme récente"), ainsi que leur nombre de HR sur ces
    10 derniers matchs. Comme pour les lanceurs, hydratation combinée
    (saison + lastXGames) en un minimum d'appels API par lots de 40 IDs.
    """
    ids = sorted({str(i) for i in batter_ids if i})
    resultats = {}
    if not ids:
        return resultats

    for debut in range(0, len(ids), 40):
        lot = ids[debut:debut + 40]
        try:
            reponse = appeler_avec_retry(
                statsapi.get, 'people',
                {'personIds': ','.join(lot), 'hydrate': 'stats(group=[hitting],type=[season,lastXGames],limit=10)'}
            )
        except Exception:
            continue
        for p in reponse.get('people', []):
            entree = {
                'nom': p.get('fullName'),
                'slg_saison': 0.0, 'obp_saison': 0.0,
                'slg_10': 0.0, 'obp_10': 0.0, 'hr_10': 0, 'matchs_10': 0,
            }
            for bloc in p.get('stats', []):
                type_nom = bloc.get('type', {}).get('displayName')
                splits = bloc.get('splits') or []
                if not splits:
                    continue
                s = splits[0].get('stat', {})
                if type_nom == 'season':
                    entree['slg_saison'] = _parser_stat_flottant(s.get('slg'))
                    entree['obp_saison'] = _parser_stat_flottant(s.get('obp'))
                elif type_nom == 'lastXGames':
                    entree['slg_10'] = _parser_stat_flottant(s.get('slg'))
                    entree['obp_10'] = _parser_stat_flottant(s.get('obp'))
                    entree['hr_10'] = int(s.get('homeRuns') or 0)
                    entree['matchs_10'] = int(s.get('gamesPlayed') or 0)
            resultats[p['id']] = entree
    return resultats


def _normaliser_colonne(serie: pd.Series) -> pd.Series:
    """
    Normalisation min-max dans [0, 1] d'une colonne de statistiques, pour pouvoir
    combiner des métriques d'échelles très différentes (ex: SLG ~0.3-0.6, HR sur 10
    matchs 0-6, ERA 2-6) dans un même indice pondéré. Renvoie une série neutre à 0.5
    si la colonne est constante (évite une division par zéro sans fausser le classement).
    """
    minimum, maximum = serie.min(), serie.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series([0.5] * len(serie), index=serie.index)
    return (serie - minimum) / (maximum - minimum)


def _calculer_top5_home_runs(candidats: list) -> pd.DataFrame:
    """
    Construit le classement "Top 5 Home Runs probables" à partir de la liste de
    candidats (un dict par batteur titulaire d'un match du jour dont la lineup est
    connue). Indice pondéré : SLG récent 45% + HR/10 derniers matchs 35% +
    HR/9 du lanceur adverse 20% (les 3 facteurs demandés), chaque métrique étant
    normalisée (min-max) sur l'ensemble des candidats du jour avant pondération.
    """
    if not candidats:
        return pd.DataFrame()
    df = pd.DataFrame(candidats)
    indice = (
        _normaliser_colonne(df['SLG récent']) * 0.45
        + _normaliser_colonne(df['HR (10 derniers matchs)']) * 0.35
        + _normaliser_colonne(df['HR/9 lanceur adverse']) * 0.20
    ) * 100
    df['Indice HR (/100)'] = indice.round(1)
    df = df.sort_values('Indice HR (/100)', ascending=False).head(5).reset_index(drop=True)
    return df[[
        'Joueur', 'Équipe', 'Adversaire', 'Lanceur adverse',
        'SLG récent', 'HR (10 derniers matchs)', 'HR/9 lanceur adverse', 'Indice HR (/100)'
    ]]


def _calculer_top5_runs(candidats: list) -> pd.DataFrame:
    """
    Construit le classement "Top 5 joueurs pour marquer un run" à partir de la liste
    de candidats. Indice pondéré : OBP 45% + bonus de position dans le lineup
    (favorise les positions 1 à 4) 25% + ERA du lanceur adverse 30%, chaque métrique
    étant normalisée (min-max) sur l'ensemble des candidats du jour avant pondération.
    """
    if not candidats:
        return pd.DataFrame()
    df = pd.DataFrame(candidats)
    # Bonus de position : décroît linéairement de la place 1 (bonus max) à la place 9
    # (bonus nul), pour "privilégier les batteurs 1 à 4" tout en restant continu.
    bonus_position = (9 - df['Position lineup']).clip(lower=0)
    indice = (
        _normaliser_colonne(df['OBP']) * 0.45
        + _normaliser_colonne(bonus_position) * 0.25
        + _normaliser_colonne(df['ERA lanceur adverse']) * 0.30
    ) * 100
    df['Indice Run (/100)'] = indice.round(1)
    df = df.sort_values('Indice Run (/100)', ascending=False).head(5).reset_index(drop=True)
    return df[[
        'Joueur', 'Équipe', 'Adversaire', 'Lanceur adverse',
        'OBP', 'Position lineup', 'ERA lanceur adverse', 'Indice Run (/100)'
    ]]


def _total_runs_predit(moyenne_home, moyenne_away):
    """
    Total de runs projeté pour un match = somme des moyennes de runs marqués par
    chaque équipe sur ses 10 derniers matchs (`obtenir_moyennes_runs_recentes_toutes_equipes`).
    Sert de projection "Over/Under" dans le bilan des prédictions de la veille (cf.
    `obtenir_ligne_over_under_saison`). Retourne None si l'une des deux moyennes n'est
    pas disponible.
    """
    if moyenne_home is None or moyenne_away is None or pd.isna(moyenne_home) or pd.isna(moyenne_away):
        return None
    return round(float(moyenne_home) + float(moyenne_away), 2)


def _top_candidats_hr_match(candidats_hr: list, equipe_nom: str, adversaire_nom: str, n: int = 2) -> list:
    """
    Les `n` joueurs les plus en forme au HR (10 derniers matchs) d'une équipe, pour un
    match donné, à partir de la liste `candidats_hr` déjà construite dans
    `construire_donnees_hot_pronostics` (filtrée par nom d'équipe ET d'adversaire pour
    rester robuste aux doubles programmes / homonymies improbables).
    """
    pertinents = [
        c for c in candidats_hr
        if c.get('Équipe') == equipe_nom and c.get('Adversaire') == adversaire_nom
    ]
    pertinents_tries = sorted(pertinents, key=lambda c: c.get('HR (10 derniers matchs)', 0), reverse=True)
    return [c['Joueur'] for c in pertinents_tries[:n]]


@st.cache_data(show_spinner=False, ttl=1800)
def construire_donnees_hot_pronostics(annee: int):
    """
    Calcul GLOBAL et coûteux (mis en cache via @st.cache_data, ttl=30min) qui scanne
    TOUS les matchs du jour et construit les 3 tableaux de l'onglet "Hot Pronostics" :
    Top 5 Home Runs, Top 5 joueurs pour marquer un run, et le récapitulatif Win/Lose
    de chaque confrontation. Ce calcul est indépendant de l'équipe sélectionnée dans
    la sidebar, donc mis en cache séparément (clé = `annee` uniquement) pour ne
    jamais être relancé inutilement quand l'utilisateur change d'équipe.

    Retourne (matchs_du_jour, df_top5_hr, df_top5_runs, df_victoires).
    """
    matchs_du_jour = obtenir_calendrier_jour_avec_lineups(annee)
    if not matchs_du_jour:
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    moyennes_runs_equipes = obtenir_moyennes_runs_recentes_toutes_equipes(annee)

    tous_pitcher_ids = set()
    tous_batter_ids = set()
    for m in matchs_du_jour:
        tous_pitcher_ids.update([m['home_pitcher_id'], m['away_pitcher_id']])
        tous_batter_ids.update(j['id'] for j in m['home_lineup'])
        tous_batter_ids.update(j['id'] for j in m['away_lineup'])

    stats_lanceurs = obtenir_stats_lanceurs_par_id(tuple(tous_pitcher_ids))
    stats_batteurs = obtenir_stats_batteurs_par_id(tuple(tous_batter_ids))

    candidats_hr = []
    candidats_runs = []
    lignes_victoire = []

    for m in matchs_du_jour:
        stats_p_home = stats_lanceurs.get(m['home_pitcher_id'])
        stats_p_away = stats_lanceurs.get(m['away_pitcher_id'])

        # --- Tableau Win/Lose : on réutilise TEL QUEL le modèle heuristique déjà
        # validé dans l'onglet "Prédictions du jour" (`predire_probabilite_victoire`),
        # mais ici avec les moyennes de runs RÉELLES des deux équipes (au lieu du
        # proxy "runs concédés par notre équipe" utilisé dans l'onglet mono-équipe,
        # où l'attaque adverse n'était pas directement disponible).
        pct_home, pct_away = predire_probabilite_victoire(
            moyennes_runs_equipes.get(m['home_id']),
            moyennes_runs_equipes.get(m['away_id']),
            stats_p_home,
            stats_p_away,
            est_domicile=True,
        )
        lignes_victoire.append({
            'Heure (France)': m['heure_paris'],
            'Équipe Domicile': m['home_name'],
            'Lanceur Domicile': m['home_pitcher_name'] or 'Non annoncé',
            'Équipe Extérieur': m['away_name'],
            'Lanceur Extérieur': m['away_pitcher_name'] or 'Non annoncé',
            'Proba Domicile (%)': pct_home,
            'Proba Extérieur (%)': pct_away,
        })

        # --- Candidats HR / Runs : chaque lineup connue est croisée avec le lanceur
        # partant ADVERSE (celui qu'elle affrontera aujourd'hui).
        for camp_lineup, lanceur_adverse, equipe_nom, adversaire_nom in (
            (m['home_lineup'], stats_p_away, m['home_name'], m['away_name']),
            (m['away_lineup'], stats_p_home, m['away_name'], m['home_name']),
        ):
            for joueur in camp_lineup:
                s = stats_batteurs.get(joueur['id'])
                if not s:
                    continue
                # SLG "récent" : sur les 10 derniers matchs si le joueur en a
                # suffisamment joué récemment (>=3), sinon repli sur le SLG saison
                # (évite qu'un retour de blessure/appel des ligues mineures avec 1
                # seul match récent fausse le classement dans un sens ou l'autre).
                slg_recent = s['slg_10'] if s['matchs_10'] >= 3 else s['slg_saison']
                nom_lanceur_adverse = lanceur_adverse['nom'] if lanceur_adverse else 'Non annoncé'

                candidats_hr.append({
                    'Joueur': joueur['nom'],
                    'Équipe': equipe_nom,
                    'Adversaire': adversaire_nom,
                    'Lanceur adverse': nom_lanceur_adverse,
                    'SLG récent': slg_recent,
                    'HR (10 derniers matchs)': s['hr_10'],
                    'HR/9 lanceur adverse': lanceur_adverse['hr_par_9'] if lanceur_adverse else 1.1,
                })
                candidats_runs.append({
                    'Joueur': joueur['nom'],
                    'Équipe': equipe_nom,
                    'Adversaire': adversaire_nom,
                    'Lanceur adverse': nom_lanceur_adverse,
                    'OBP': s['obp_saison'],
                    'Position lineup': joueur['position'],
                    'ERA lanceur adverse': lanceur_adverse['era'] if lanceur_adverse else 4.5,
                })

    df_top5_hr = _calculer_top5_home_runs(candidats_hr)
    df_top5_runs = _calculer_top5_runs(candidats_runs)
    df_victoires = pd.DataFrame(lignes_victoire)

    # --- Archivage de l'instantané du jour (pour le "Bilan des Prédictions" de la
    # veille, onglet Résumé, cf. `_sauvegarder_predictions_du_jour`) : on ne conserve
    # que ce qui est nécessaire à une comparaison ultérieure avec le résultat réel une
    # fois le match terminé (probabilité de victoire, total de runs projeté pour les
    # deux équipes, et candidats HR les plus en forme de chaque équipe). Indexé par
    # `game_id` (identifiant unique statsapi), plus fiable qu'un nom d'équipe en cas
    # de double programme (doubleheader).
    matches_snapshot = [
        {
            'game_id': m.get('game_id'),
            'home_name': m['home_name'],
            'away_name': m['away_name'],
            'proba_home': ligne_victoire.get('Proba Domicile (%)'),
            'proba_away': ligne_victoire.get('Proba Extérieur (%)'),
            'total_runs_predit': _total_runs_predit(
                moyennes_runs_equipes.get(m['home_id']), moyennes_runs_equipes.get(m['away_id'])
            ),
            'candidats_hr_home': _top_candidats_hr_match(candidats_hr, m['home_name'], m['away_name']),
            'candidats_hr_away': _top_candidats_hr_match(candidats_hr, m['away_name'], m['home_name']),
        }
        for m, ligne_victoire in zip(matchs_du_jour, lignes_victoire)
    ]
    _sauvegarder_predictions_du_jour(datetime.now(TZ_US_EASTERN).strftime('%Y-%m-%d'), matches_snapshot)

    return matchs_du_jour, df_top5_hr, df_top5_runs, df_victoires


# ============================================================
# 3 ter. ONGLET "RÉSUMÉ" - Scores en direct et terminés du jour
# ============================================================
# Ce bloc alimente le tout premier onglet de l'application : un tableau récapitulatif
# de TOUS les matchs MLB du jour (à venir / en cours / terminés), avec un bouton de
# rafraîchissement manuel qui ne recharge QUE cet onglet (via `st.fragment`), pas toute
# la page. Il réutilise le modèle de prédiction déjà calculé pour "Hot Pronostics"
# (`construire_donnees_hot_pronostics`) pour la colonne "Comparatif Prédiction", au lieu
# de dupliquer le calcul de probabilité de victoire.

@st.cache_data
def get_team_id_vers_infos(year: int = None):
    """
    Construit un dictionnaire {team_id: {'abbr', 'nom_complet', 'nickname'}} pour
    l'année donnée. `nickname` (ex: 'Yankees') vient du champ `teamName` de statsapi -
    distinct de `name` (nom complet, ex: 'New York Yankees') déjà utilisé ailleurs dans
    ce fichier. Indexé par `team_id` (et non par abréviation) car les matchs du jour
    (`statsapi.schedule`) exposent directement `home_id`/`away_id`, ce qui évite tout
    risque d'ambiguïté de correspondance par nom d'équipe.
    """
    if year is None:
        year = ANNEE_COURANTE
    reponse = appeler_avec_retry(statsapi.get, 'teams', {'sportIds': 1, 'season': year})
    result = {}
    for t in reponse['teams']:
        result[t['id']] = {
            'abbr': t.get('abbreviation', '?'),
            'nom_complet': t.get('name', '?'),
            'nickname': t.get('teamName') or t.get('name', '?'),
        }
    return result


def _ordinal_anglais(n) -> str:
    """Formate un entier en ordinal anglais (1 -> '1st', 4 -> '4th', 11 -> '11th', ...)."""
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffixe = 'th'
    else:
        suffixe = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffixe}"


def _formater_statut_match(status_brut: str, current_inning, inning_state: str) -> str:
    """
    Traduit le statut brut statsapi (ex: 'In Progress', 'Final', 'Scheduled') en l'une
    des catégories demandées : 'À venir', 'En cours (Bot 4th)' ou 'Terminé' - avec un
    repli explicite pour les statuts rares (reporté/suspendu/annulé) plutôt que de les
    faire tomber silencieusement dans une mauvaise catégorie.
    """
    s_lower = (status_brut or '').strip().lower()

    if 'final' in s_lower or 'game over' in s_lower:
        return "Terminé"

    if s_lower == 'in progress':
        abrev_manche = {
            'top': 'Top', 'bottom': 'Bot', 'middle': 'Mid', 'end': 'End',
        }.get((inning_state or '').strip().lower(), (inning_state or '').strip())
        try:
            manche_str = _ordinal_anglais(current_inning)
        except (TypeError, ValueError):
            manche_str = str(current_inning or '').strip()
        detail = f"{abrev_manche} {manche_str}".strip()
        return f"En cours ({detail})" if detail else "En cours"

    if 'postponed' in s_lower:
        return "Reporté"
    if 'suspended' in s_lower:
        return "Suspendu"
    if 'cancelled' in s_lower or 'canceled' in s_lower:
        return "Annulé"

    return "À venir"  # Scheduled, Pre-Game, Warmup, Delayed Start, ...


@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def obtenir_hr_joueurs_match(game_id: int, est_domicile: bool, cache_bust: int = 0):
    """
    Récupère, via le boxscore statsapi d'un match, la liste des home runs marqués par
    chaque joueur d'une équipe (domicile ou extérieur) sous forme de tuples
    (nom_joueur, nb_hr). Fonction dédiée à l'onglet "Résumé" (plutôt que de réutiliser
    `get_stats_offensives_match`, partagée avec l'onglet "Analyse par Équipe" et jamais
    invalidée) car ici le match peut être EN COURS : `cache_bust` change la clé de
    cache Streamlit à la demande (incrémenté par le bouton "Rafraîchir"), ce qui permet
    de forcer un nouvel appel réseau sans dépendre d'un simple TTL. Le paramètre n'est
    jamais lu dans le corps de la fonction, il ne sert qu'à invalider le cache.
    `ttl=3600` reste un filet de sécurité pour éviter une croissance illimitée du cache,
    pas le mécanisme principal de fraîcheur des données.
    """
    if not game_id:
        return []
    try:
        box = appeler_avec_retry(statsapi.boxscore_data, int(game_id))
        batters = box.get('homeBatters', []) if est_domicile else box.get('awayBatters', [])
        resultats = []
        for b in batters:
            if not b.get('personId'):
                continue  # ligne d'en-tête du tableau, pas un joueur
            try:
                hr = int(b.get('hr', 0) or 0)
            except (ValueError, TypeError):
                hr = 0
            if hr > 0:
                resultats.append((b.get('name', 'Inconnu'), hr))
        return resultats
    except Exception:
        # Ne doit jamais faire planter l'onglet Résumé : simplement pas de HR affiché.
        return []


def _formater_segment_hr(abbr: str, hr_liste: list) -> str:
    """Formate les HR d'UNE équipe : 'NYY: 2 (Judge, Soto)' ou 'NYY: 0' si aucun HR."""
    total = sum(hr for _, hr in hr_liste)
    if total <= 0:
        return f"{abbr}: 0"
    noms = [nom if hr <= 1 else f"{nom} x{hr}" for nom, hr in hr_liste]
    return f"{abbr}: {total} ({', '.join(noms)})"


def _formater_cellule_hr(away_abbr: str, hr_away: list, home_abbr: str, hr_home: list) -> str:
    """Combine les HR des deux équipes d'un match dans une seule cellule de tableau."""
    return f"{_formater_segment_hr(away_abbr, hr_away)} | {_formater_segment_hr(home_abbr, hr_home)}"


def _comparer_prediction_vs_score(pred, home_nick: str, away_nick: str, home_score: int, away_score: int, a_commence: bool):
    """
    Retourne (texte_comparatif, icone_resultat) pour la colonne "Résultat vs Algo".
    - `pred` : ligne (pandas Series) issue de `df_victoires` (Hot Pronostics) pour ce
      match, ou None si aucune prédiction n'est encore disponible (lineups/lanceurs
      partants pas encore publiés) -> ("Non disponible", "⏳").
    - Sinon : l'équipe favorite est celle avec la probabilité de victoire la plus
      haute. On compare cette équipe favorite à l'équipe actuellement en tête (ou
      gagnante si le match est terminé) : ✅ si elle mène/a gagné, ❌ si elle est
      menée/a perdu, ⏳ si le match n'a pas commencé ou si le score est à égalité.
    """
    if pred is None:
        return "Non disponible", "⏳"

    pct_home = pred.get('Proba Domicile (%)')
    pct_away = pred.get('Proba Extérieur (%)')
    if pct_home is None or pct_away is None or pd.isna(pct_home) or pd.isna(pct_away):
        return "Non disponible", "⏳"

    equipe_favorite = home_nick if pct_home >= pct_away else away_nick
    pct_favori = max(pct_home, pct_away)
    comparatif = f"{equipe_favorite} à {pct_favori:.0f}%"

    if not a_commence or home_score == away_score:
        return comparatif, "⏳"

    equipe_en_tete = home_nick if home_score > away_score else away_nick
    icone = "✅" if equipe_en_tete == equipe_favorite else "❌"
    return comparatif, icone


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def construire_resume_matchs_du_jour(annee: int, cache_bust: int = 0):
    """
    Construit le tableau récapitulatif de TOUS les matchs MLB du jour (à venir, en
    cours, terminés) pour l'onglet "Résumé". `cache_bust` sert uniquement à invalider
    le cache Streamlit à la demande (bouton "Rafraîchir les scores en direct") - le
    calcul du modèle de prédiction ("Hot Pronostics") n'est PAS reproduit à chaque
    rafraîchissement (il a son propre cache à `ttl=1800`, car il ne change pas au fil
    du match), seuls les scores/statuts/HR en direct sont re-récupérés.

    Retourne (DataFrame, message_erreur). En cas d'échec réseau, le DataFrame est vide
    et `message_erreur` contient un texte à afficher via `st.error` - aucune exception
    ne remonte jamais à l'appelant (l'application ne doit jamais planter à cause d'un
    appel API en direct).
    """
    if annee != ANNEE_COURANTE:
        return pd.DataFrame(), None

    aujourdhui_us = datetime.now(TZ_US_EASTERN).strftime('%Y-%m-%d')
    try:
        matchs_live = appeler_avec_retry(statsapi.schedule, date=aujourdhui_us, sportId=1)
    except Exception as e:
        return pd.DataFrame(), (
            f"Impossible de récupérer les scores en direct pour le moment ({e}). "
            "Réessayez dans quelques instants avec le bouton de rafraîchissement."
        )

    if not matchs_live:
        return pd.DataFrame(), None

    try:
        infos_equipes = get_team_id_vers_infos(annee)
    except Exception:
        infos_equipes = {}

    # Prédictions déjà calculées pour "Hot Pronostics" (même modèle, même journée),
    # réutilisées ici pour la colonne "Comparatif Prédiction" - alignées par game_id
    # (les deux fonctions parcourent le même calendrier du jour, dans le même ordre,
    # mais on indexe explicitement par game_id pour rester robuste à tout changement
    # d'ordre entre les deux appels).
    try:
        matchs_lineups, _, _, df_victoires = construire_donnees_hot_pronostics(annee)
    except Exception:
        matchs_lineups, df_victoires = [], pd.DataFrame()

    predictions_par_game_id = {}
    for idx, m in enumerate(matchs_lineups):
        if idx < len(df_victoires):
            predictions_par_game_id[m.get('game_id')] = df_victoires.iloc[idx]

    lignes = []
    for g in matchs_live:
        game_id = g.get('game_id')
        info_home = infos_equipes.get(g.get('home_id'), {})
        info_away = infos_equipes.get(g.get('away_id'), {})
        home_nick = info_home.get('nickname') or g.get('home_name') or '?'
        away_nick = info_away.get('nickname') or g.get('away_name') or '?'
        home_abbr = info_home.get('abbr') or (home_nick[:3].upper() if home_nick else '???')
        away_abbr = info_away.get('abbr') or (away_nick[:3].upper() if away_nick else '???')

        statut_str = _formater_statut_match(g.get('status'), g.get('current_inning'), g.get('inning_state'))
        a_commence = statut_str == "Terminé" or statut_str.startswith("En cours") or statut_str == "Suspendu"

        try:
            home_score = int(g.get('home_score') or 0)
            away_score = int(g.get('away_score') or 0)
        except (TypeError, ValueError):
            home_score, away_score = 0, 0

        if a_commence:
            score_str = f"{away_abbr} {away_score} - {home_abbr} {home_score}"
            # Colonne texte (pas numérique) volontairement : elle doit pouvoir afficher
            # "—" pour les matchs pas encore commencés sans faire planter la sérialisation
            # Arrow du tableau (colonne à types mixtes int/str sinon).
            total_runs = str(home_score + away_score)
            hr_home = obtenir_hr_joueurs_match(game_id, True, cache_bust)
            hr_away = obtenir_hr_joueurs_match(game_id, False, cache_bust)
            hr_str = _formater_cellule_hr(away_abbr, hr_away, home_abbr, hr_home)
        else:
            score_str = "—"
            total_runs = "—"
            hr_str = "—"

        pred = predictions_par_game_id.get(game_id)
        comparatif_str, resultat_icone = _comparer_prediction_vs_score(
            pred, home_nick, away_nick, home_score, away_score, a_commence
        )

        lignes.append({
            'Match': f"{away_nick} vs {home_nick}",
            'Statut': statut_str,
            'Score': score_str,
            'Total Runs': total_runs,
            'Home Runs': hr_str,
            'Comparatif Prédiction': comparatif_str,
            'Résultat vs Algo': resultat_icone,
        })

    return pd.DataFrame(lignes), None


# ------------------------------------------------------------------------------
# BILAN DES PRÉDICTIONS DE LA VEILLE (menu déroulant en tête de l'onglet "Résumé")
# ------------------------------------------------------------------------------
# statsapi ne publie aucune ligne de paris officielle (contrairement aux sites de
# paris sportifs) : à défaut, la "ligne" Over/Under utilisée ci-dessous pour qualifier
# un match de "à forte marque" (Over) ou "à faible marque" (Under) est la moyenne
# réelle de runs cumulés (les deux équipes confondues) sur tous les matchs déjà joués
# cette saison - la référence la plus neutre et la plus objective disponible sans
# source de paris tierce.
@st.cache_data(show_spinner=False, ttl=3600)
def obtenir_ligne_over_under_saison(annee: int) -> float:
    """
    Moyenne de runs totaux (2 équipes cumulées) sur tous les matchs MLB déjà joués
    cette saison, tous équipes confondues - sert de ligne de référence Over/Under pour
    le bilan des prédictions de la veille. Repli à 8.5 (ordre de grandeur usuel en MLB)
    si aucune donnée n'est encore disponible (tout début de saison).
    """
    date_debut = f"{annee}-03-01"
    date_fin = datetime.now(TZ_US_EASTERN).strftime('%Y-%m-%d')
    try:
        matchs = appeler_avec_retry(
            statsapi.schedule, start_date=date_debut, end_date=date_fin, sportId=1
        )
    except Exception:
        return 8.5

    totaux = []
    for g in matchs:
        if g.get('status') != 'Final':
            continue
        home_score, away_score = g.get('home_score'), g.get('away_score')
        if home_score is None or away_score is None:
            continue
        try:
            totaux.append(int(home_score) + int(away_score))
        except (TypeError, ValueError):
            continue

    if not totaux:
        return 8.5
    return round(sum(totaux) / len(totaux), 2)


def _formater_vainqueur(nom_home: str, nom_away: str, home_score: int, away_score: int) -> str:
    """Nom de l'équipe gagnante, ou 'Match nul' (cas très rare en MLB - matchs suspendus/annulés à égalité)."""
    if home_score == away_score:
        return "Match nul"
    return nom_home if home_score > away_score else nom_away


def _bilan_victoire(proba_home, proba_away, nom_home: str, nom_away: str, home_score: int, away_score: int):
    """Retourne (texte, icône) comparant l'équipe favorite annoncée hier à la gagnante réelle."""
    if proba_home is None or proba_away is None or pd.isna(proba_home) or pd.isna(proba_away):
        return "Prédiction non disponible", "⏳"
    if home_score == away_score:
        return "Match nul (pas de favori confirmé)", "⏳"
    favori = nom_home if proba_home >= proba_away else nom_away
    pct_favori = max(proba_home, proba_away)
    gagnant = nom_home if home_score > away_score else nom_away
    icone = "✅" if favori == gagnant else "❌"
    return f"{favori} favori à {pct_favori:.0f}% → vainqueur : {gagnant}", icone


def _bilan_over_under(total_runs_predit, total_runs_reel: int, ligne: float):
    """Retourne (texte, icône) comparant la projection Over/Under d'hier au total réel."""
    if total_runs_predit is None:
        return "Prédiction non disponible", "⏳"
    direction_predite = "Over" if total_runs_predit > ligne else "Under"
    direction_reelle = "Over" if total_runs_reel > ligne else "Under"
    icone = "✅" if direction_predite == direction_reelle else "❌"
    return (
        f"{direction_predite} annoncé (projection {total_runs_predit:.1f}, ligne {ligne:.1f}) "
        f"→ réel {total_runs_reel} ({direction_reelle})"
    ), icone


def _bilan_hr(candidats_home: list, candidats_away: list, hr_home_reels: list, hr_away_reels: list):
    """Retourne (texte, icône) : au moins un des joueurs surveillés hier a-t-il réellement frappé un HR ?"""
    candidats = [c for c in (candidats_home or []) + (candidats_away or []) if c]
    if not candidats:
        return "Prédiction non disponible", "⏳"
    scoreurs_reels = {nom for nom, _ in hr_home_reels} | {nom for nom, _ in hr_away_reels}
    touches = [c for c in candidats if c in scoreurs_reels]
    icone = "✅" if touches else "❌"
    texte = f"Surveillés : {', '.join(candidats)}"
    if touches:
        texte += f" → a frappé : {', '.join(touches)}"
    return texte, icone


@st.cache_data(show_spinner=False, ttl=3600)
def construire_bilan_veille(annee: int):
    """
    Construit le tableau "Résultats de la veille et Bilan des Prédictions" : reprend
    la structure du tableau des matchs du jour (`construire_resume_matchs_du_jour`),
    mais pour la date d'HIER (heure US, Est) et avec les matchs forcément terminés,
    enrichi de colonnes de bilan comparant la prédiction sauvegardée hier
    (`_sauvegarder_predictions_du_jour`, appelée automatiquement depuis
    `construire_donnees_hot_pronostics`) au résultat réel.

    Comme cette fonction n'est appelée QUE lorsque l'utilisateur ouvre le menu
    déroulant (cf. `afficher_bilan_predictions_veille`), elle n'a aucun coût au
    chargement initial de l'onglet "Résumé".

    Retourne (DataFrame, message_erreur, predictions_disponibles) :
      - `predictions_disponibles` (bool) indique si UN AU MOINS instantané de
        prédictions a été retrouvé pour la date d'hier - utilisé par
        `afficher_bilan_predictions_veille` pour distinguer "aucune prédiction n'a
        jamais été archivée pour cette date" (cas normal les tout premiers jours après
        l'ajout de cette fonctionnalité, ou si l'app n'a pas été ouverte la veille) du
        cas où le tableau est simplement vide pour une autre raison.
    Sur le même modèle que `construire_resume_matchs_du_jour`, aucune exception n'est
    jamais remontée à l'appelant.
    """
    hier_us = datetime.now(TZ_US_EASTERN) - timedelta(days=1)
    date_hier_str = hier_us.strftime('%Y-%m-%d')

    try:
        matchs_hier = appeler_avec_retry(statsapi.schedule, date=date_hier_str, sportId=1)
    except Exception as e:
        return pd.DataFrame(), (
            f"Impossible de récupérer les résultats d'hier pour le moment ({e}). "
            "Réessayez en rouvrant ce menu dans quelques instants."
        ), True

    if not matchs_hier:
        return pd.DataFrame(), None, True

    try:
        infos_equipes = get_team_id_vers_infos(annee)
    except Exception:
        infos_equipes = {}

    predictions_hier = _charger_historique_predictions().get(date_hier_str, {}).get('matches', [])
    predictions_disponibles = len(predictions_hier) > 0
    predictions_par_game_id = {p.get('game_id'): p for p in predictions_hier}

    ligne_ou = obtenir_ligne_over_under_saison(annee)

    lignes = []
    for g in matchs_hier:
        if g.get('status') != 'Final':
            continue
        game_id = g.get('game_id')
        info_home = infos_equipes.get(g.get('home_id'), {})
        info_away = infos_equipes.get(g.get('away_id'), {})
        home_nick = info_home.get('nickname') or g.get('home_name') or '?'
        away_nick = info_away.get('nickname') or g.get('away_name') or '?'
        home_abbr = info_home.get('abbr') or (home_nick[:3].upper() if home_nick else '???')
        away_abbr = info_away.get('abbr') or (away_nick[:3].upper() if away_nick else '???')

        try:
            home_score = int(g.get('home_score'))
            away_score = int(g.get('away_score'))
        except (TypeError, ValueError):
            continue
        total_reel = home_score + away_score

        hr_home = obtenir_hr_joueurs_match(game_id, True)
        hr_away = obtenir_hr_joueurs_match(game_id, False)

        pred = predictions_par_game_id.get(game_id)
        candidats_hr_home = pred.get('candidats_hr_home', []) if pred else []
        candidats_hr_away = pred.get('candidats_hr_away', []) if pred else []
        proba_home = pred.get('proba_home') if pred else None
        proba_away = pred.get('proba_away') if pred else None
        total_predit = pred.get('total_runs_predit') if pred else None

        texte_victoire, icone_victoire = _bilan_victoire(
            proba_home, proba_away, home_nick, away_nick, home_score, away_score
        )
        texte_ou, icone_ou = _bilan_over_under(total_predit, total_reel, ligne_ou)
        texte_hr, icone_hr = _bilan_hr(candidats_hr_home, candidats_hr_away, hr_home, hr_away)

        lignes.append({
            'Match': f"{away_nick} vs {home_nick}",
            'Statut': "Terminé",
            'Score': f"{away_abbr} {away_score} - {home_abbr} {home_score}",
            'Total Runs': str(total_reel),
            'Home Runs': _formater_cellule_hr(away_abbr, hr_away, home_abbr, hr_home),
            'Vainqueur': _formater_vainqueur(home_nick, away_nick, home_score, away_score),
            'Victoire prédite': texte_victoire,
            'Over/Under prédit': texte_ou,
            'HR surveillés': texte_hr,
            'Bilan': f"Victoire {icone_victoire} · Over/Under {icone_ou} · HR {icone_hr}",
        })

    return pd.DataFrame(lignes), None, predictions_disponibles


def afficher_bilan_predictions_veille(annee: int):
    """
    Corps du menu déroulant "📅 Résultats de la veille et Bilan des Prédictions" :
    appelé uniquement quand ce menu est ouvert (cf. garde `expander.open` dans
    `afficher_onglet_resume`), donc sans coût réseau tant que l'utilisateur ne l'a
    pas déplié.
    """
    if annee != ANNEE_COURANTE:
        st.info(
            f"Le bilan de la veille n'est disponible que pour la saison en cours "
            f"({ANNEE_COURANTE})."
        )
        return

    with st.spinner("Récupération des résultats d'hier et calcul du bilan des prédictions..."):
        df_bilan, message_erreur, predictions_disponibles = construire_bilan_veille(annee)

    if message_erreur:
        st.error(f"⚠️ {message_erreur}")
        return

    if df_bilan.empty:
        st.info("Aucun match MLB terminé hier (heure US, Est).")
        return

    if not predictions_disponibles:
        st.info(
            "ℹ️ Aucune prédiction n'a été archivée hier pour ces matchs, donc les colonnes de "
            "bilan ci-dessous affichent \"Prédiction non disponible\" - les résultats réels, eux, "
            "sont bien à jour. Cela arrive si l'application n'a pas été consultée du tout hier "
            "(l'archivage se fait uniquement à l'ouverture de l'onglet Résumé ou Hot Pronostics), "
            "ou si cette fonctionnalité vient tout juste d'être ajoutée : le bilan se remplira "
            "automatiquement à partir de demain."
        )

    st.dataframe(
        df_bilan,
        column_config={
            "Match": st.column_config.TextColumn("Match", width="medium"),
            "Statut": st.column_config.TextColumn("Statut", width="small"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Total Runs": st.column_config.TextColumn("Total Runs", width="small"),
            "Home Runs": st.column_config.TextColumn("Home Runs", width="large"),
            "Vainqueur": st.column_config.TextColumn("Vainqueur", width="medium"),
            "Victoire prédite": st.column_config.TextColumn("Victoire prédite", width="large"),
            "Over/Under prédit": st.column_config.TextColumn("Over/Under prédit", width="large"),
            "HR surveillés": st.column_config.TextColumn("HR surveillés", width="large"),
            "Bilan": st.column_config.TextColumn("Bilan", width="large"),
        },
        hide_index=True,
    )

    st.caption(
        "**Méthodologie** — Victoire : ✅ si l'équipe favorite (probabilité la plus haute) a "
        "réellement gagné. Over/Under : ligne de référence = moyenne réelle de runs cumulés par "
        "match sur la saison en cours ; ✅ si notre projection (moyenne de runs des 10 derniers "
        "matchs des deux équipes) était du même côté de cette ligne que le résultat réel. HR : ✅ "
        "si au moins un des joueurs les plus en forme au HR (10 derniers matchs) de chaque équipe "
        "a effectivement frappé un home run dans ce match. ⏳ = aucune prédiction n'avait été "
        "archivée pour ce match (application non consultée la veille) ou match nul. Les "
        "prédictions ne sont archivées qu'au moment où l'onglet Résumé ou Hot Pronostics est "
        "consulté ce jour-là (pas de calcul en tâche de fond)."
    )


@st.fragment
def afficher_onglet_resume(annee: int):
    """
    Corps de l'onglet "Résumé" (menu déroulant "Bilan de la veille" + bouton de
    rafraîchissement + tableau du jour), encapsulé dans un `st.fragment` : cliquer sur
    le bouton, ou ouvrir/fermer le menu déroulant, ne relance QUE cette fonction, sans
    recharger le reste de l'application (sidebar, autres onglets) ni la page web
    entière.
    """
    # --- Menu déroulant "Bilan des Prédictions" de la veille, tout en haut de
    # l'onglet, au-dessus du tableau des matchs du jour. `on_change="rerun"` rend la
    # propriété `.open` dynamique (True/False selon l'état du menu) : le contenu
    # (requête réseau statsapi incluse) n'est donc calculé QUE si l'utilisateur a
    # effectivement déplié le menu, jamais au chargement initial de l'onglet.
    expander_veille = st.expander(
        "📅 Résultats de la veille et Bilan des Prédictions", on_change="rerun"
    )
    if expander_veille.open:
        with expander_veille:
            afficher_bilan_predictions_veille(annee)

    st.markdown("---")

    if 'resume_cache_bust' not in st.session_state:
        st.session_state.resume_cache_bust = 0
    if 'resume_derniere_actualisation' not in st.session_state:
        st.session_state.resume_derniere_actualisation = None

    col_bouton, col_info = st.columns([1, 2])
    with col_bouton:
        if st.button("🔄 Rafraîchir les scores en direct"):
            st.session_state.resume_cache_bust += 1
            st.session_state.resume_derniere_actualisation = datetime.now(TZ_PARIS)

    with col_info:
        if st.session_state.resume_derniere_actualisation:
            st.caption(
                "Dernière actualisation manuelle : "
                f"{st.session_state.resume_derniere_actualisation.strftime('%H:%M:%S')} (heure française)."
            )
        else:
            st.caption("Cliquez sur le bouton pour actualiser les scores en direct.")

    if annee != ANNEE_COURANTE:
        st.info(
            f"Le résumé du jour n'est disponible que pour la saison en cours "
            f"({ANNEE_COURANTE}). Sélectionnez {ANNEE_COURANTE} dans le menu de gauche."
        )
        return

    with st.spinner("Récupération des scores en direct..."):
        df_resume, message_erreur = construire_resume_matchs_du_jour(
            annee, st.session_state.resume_cache_bust
        )

    if message_erreur:
        st.error(f"⚠️ {message_erreur}")

    if df_resume.empty:
        if message_erreur is None:
            st.info("Aucun match n'est prévu aujourd'hui (heure US).")
        return

    st.dataframe(
        df_resume,
        column_config={
            "Match": st.column_config.TextColumn("Match", width="medium"),
            "Statut": st.column_config.TextColumn("Statut", width="small"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Total Runs": st.column_config.TextColumn("Total Runs", width="small"),
            "Home Runs": st.column_config.TextColumn("Home Runs", width="large"),
            "Comparatif Prédiction": st.column_config.TextColumn("Comparatif Prédiction", width="medium"),
            "Résultat vs Algo": st.column_config.TextColumn("Résultat vs Algo", width="small"),
        },
        hide_index=True,
    )

    st.caption(
        "✅ = l'équipe favorite de notre algorithme mène ou a gagné · ❌ = elle est menée ou a "
        "perdu · ⏳ = match pas encore commencé, à égalité, ou prédiction pas encore disponible. "
        "Le score, le total de runs et les home runs ne sont affichés qu'une fois le match "
        "commencé."
    )


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
    "📊 Résumé",
    "🔥 Hot Pronostics",
    "📊 Analyse par Équipe",
    "🔮 Prédictions du jour"
], on_change="rerun")

# --------------------------------------------------------------
# ONGLET 0: RÉSUMÉ (scores en direct et terminés du jour)
# --------------------------------------------------------------
with onglets[0]:
    if onglets[0].open:
        st.header("📊 Résumé du jour")
        st.markdown("### Suivi en direct de toutes les confrontations MLB du jour")
        afficher_onglet_resume(annee)

# --------------------------------------------------------------
# ONGLET 1: HOT PRONOSTICS (scan global de tous les matchs du jour)
# --------------------------------------------------------------
with onglets[1]:
    if onglets[1].open:
        st.header("🔥 Hot Pronostics du jour")
        st.markdown("### Les meilleurs pronostics du jour, tous matchs confondus")
        st.caption(
            "⚠️ Estimations statistiques automatiques calculées à partir des lineups probables "
            "(quand elles sont déjà publiées par les équipes), des lanceurs partants et de la "
            "forme récente des joueurs. Ce ne sont pas des garanties de résultat : simples "
            "heuristiques, à utiliser uniquement à titre informatif, avec discernement si vous "
            "vous en servez pour parier."
        )

        if annee != ANNEE_COURANTE:
            st.info(
                f"Les Hot Pronostics ne sont disponibles que pour la saison en cours "
                f"({ANNEE_COURANTE}). Sélectionnez {ANNEE_COURANTE} dans le menu de gauche."
            )
        else:
            with st.spinner("Analyse de tous les matchs du jour (lineups, lanceurs, forme récente)..."):
                matchs_jour, df_top5_hr, df_top5_runs, df_victoires = construire_donnees_hot_pronostics(annee)

            if not matchs_jour:
                st.info("Aucun match n'est prévu aujourd'hui (heure US).")
            else:
                nb_lineups_home = sum(1 for m in matchs_jour if m['home_lineup'])
                nb_lineups_away = sum(1 for m in matchs_jour if m['away_lineup'])
                nb_matchs_avec_lineup = sum(1 for m in matchs_jour if m['home_lineup'] or m['away_lineup'])
                st.caption(
                    f"📅 {len(matchs_jour)} match(s) au programme aujourd'hui (heure US) · "
                    f"lineup officielle publiée pour {nb_matchs_avec_lineup} match(s) sur {len(matchs_jour)} "
                    "(les lineups sont généralement annoncées 1 à 3h avant chaque match - "
                    "revenez plus tard pour voir apparaître les matchs restants)."
                )

                st.markdown("---")
                st.subheader("💣 Top 5 Home Runs probables")
                if df_top5_hr.empty:
                    st.info(
                        "Aucune lineup officielle n'est encore publiée pour les matchs du jour. "
                        "Réessayez plus près de l'heure des matchs."
                    )
                else:
                    st.dataframe(
                        df_top5_hr,
                        column_config={
                            "SLG récent": st.column_config.NumberColumn("SLG récent", format="%.3f"),
                            "HR (10 derniers matchs)": st.column_config.NumberColumn("HR (10 derniers matchs)", format="%d"),
                            "HR/9 lanceur adverse": st.column_config.NumberColumn("HR/9 lanceur adverse", format="%.2f"),
                            "Indice HR (/100)": st.column_config.ProgressColumn(
                                "Indice HR (/100)", min_value=0, max_value=100, format="%.0f"
                            ),
                        },
                        hide_index=True,
                    )

                st.markdown("---")
                st.subheader("🏃 Top 5 joueurs pour marquer un run")
                if df_top5_runs.empty:
                    st.info(
                        "Aucune lineup officielle n'est encore publiée pour les matchs du jour. "
                        "Réessayez plus près de l'heure des matchs."
                    )
                else:
                    st.dataframe(
                        df_top5_runs,
                        column_config={
                            "OBP": st.column_config.NumberColumn("OBP", format="%.3f"),
                            "Position lineup": st.column_config.NumberColumn("Position lineup", format="%d"),
                            "ERA lanceur adverse": st.column_config.NumberColumn("ERA lanceur adverse", format="%.2f"),
                            "Indice Run (/100)": st.column_config.ProgressColumn(
                                "Indice Run (/100)", min_value=0, max_value=100, format="%.0f"
                            ),
                        },
                        hide_index=True,
                    )

                st.markdown("---")
                st.subheader("🎲 Probabilités Win/Lose du jour")
                if df_victoires.empty:
                    st.info("Aucune donnée de probabilité de victoire disponible pour le moment.")
                else:
                    st.dataframe(
                        df_victoires,
                        column_config={
                            "Proba Domicile (%)": st.column_config.ProgressColumn(
                                "Proba Domicile (%)", min_value=0, max_value=100, format="%.1f%%"
                            ),
                            "Proba Extérieur (%)": st.column_config.ProgressColumn(
                                "Proba Extérieur (%)", min_value=0, max_value=100, format="%.1f%%"
                            ),
                        },
                        hide_index=True,
                    )

                st.caption(
                    "**Méthodologie** — Home Runs : SLG récent (45%) + HR sur les 10 derniers matchs "
                    "(35%) + HR/9 du lanceur partant adverse (20%). Runs : OBP saison (45%) + position "
                    "dans le lineup (25%, positions 1 à 4 favorisées) + ERA du lanceur partant adverse "
                    "(30%). Win/Lose : moyenne de runs marqués sur les 10 derniers matchs de chaque "
                    "équipe + ERA/WHIP des lanceurs partants du jour (même modèle que l'onglet "
                    "\"Prédictions du jour\", détaillé plus bas). Chaque indice est normalisé sur "
                    "l'ensemble des candidats du jour, donc relatif à la journée en cours."
                )

# --------------------------------------------------------------
# ONGLET 2: ANALYSE PAR ÉQUIPE
# --------------------------------------------------------------
with onglets[2]:
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

    # --------------------------------------------------------------------
    # Graphique "Tendance des Runs par match (score équipe)" retiré pour
    # épurer l'onglet et gagner de la place. Les autres éléments (moyenne
    # de runs par match, sluggers récurrents, etc.) restent inchangés
    # ci-dessous. Voir l'historique git pour retrouver le code d'origine
    # si besoin de le réintroduire.
    # --------------------------------------------------------------------

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
# ONGLET 3: PRÉDICTIONS DU JOUR
# --------------------------------------------------------------
with onglets[3]:
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

            # --------------------------------------------------------------
            # RECOMMANDATION DE PARI OPTIMISÉE
            # --------------------------------------------------------------
            # Calculées ici (plutôt que dans leurs modules respectifs plus bas) pour
            # pouvoir alimenter la recommandation juste en dessous de la ligne
            # principale de prédiction (probabilité de victoire) ; les modules
            # "Prédiction des Runs" et "Prédiction des Joueurs" plus bas réutilisent
            # directement ces mêmes résultats (pas de recalcul, ni d'appel réseau
            # supplémentaire - ce sont de simples fonctions locales).
            prediction_runs = (
                predire_runs_match(moyenne_runs_10, moyenne_ra_10, stats_lanceur_adverse)
                if moyenne_runs_10 is not None else None
            )
            joueurs_a_surveiller = predire_joueurs_du_jour(
                cumul_runs_10, cumul_hr_10, stats_lanceur_adverse, top_n=3
            )

            conseils_paris = generer_recommandation_pari(
                pct_nous,
                pct_adverse,
                stats_lanceur_nous,
                stats_lanceur_adverse,
                prediction_runs,
                joueurs_a_surveiller,
                ligue=detecter_ligue_match(match_du_jour),
            )
            if conseils_paris:
                st.info(
                    "**💡 Recommandation de Pari Optimisée**\n\n"
                    + "\n\n".join(conseils_paris)
                )

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

            # --------------------------------------------------------------
            # VALUE BET DETECTOR (cotes de marché vs notre probabilité algorithmique)
            # --------------------------------------------------------------
            # Les cotes sont récupérées AVANT d'afficher le sous-titre, pour que celui-ci
            # cite le bookmaker RÉELLEMENT utilisé (Winamax n'est que le bookmaker
            # prioritaire - voir `ODDS_API_BOOKMAKER_PRINCIPAL` - il peut être remplacé
            # par un autre bookmaker EU si Winamax ne couvre pas ce match précis).
            st.markdown("---")

            cle_odds_api = _lire_cle_odds_api()
            cotes_match = None
            if cle_odds_api:
                cotes_du_jour = obtenir_cotes_moneyline_du_jour(ODDS_API_SPORT_KEY, cle_odds_api)
                nom_notre_equipe = EQUIPES_MLB.get(equipe_abbr, equipe_abbr)
                cotes_match = trouver_cote_du_match(
                    cotes_du_jour, nom_notre_equipe, match_du_jour['adversaire']
                )
                if cotes_match and not (cotes_match.get('cote_nous') and cotes_match.get('cote_adverse')):
                    cotes_match = None

            titre_bookmaker = f"(vs {cotes_match['bookmaker']})" if cotes_match else "(vs Winamax)"
            st.subheader(f"💰 Value Bet Detector {titre_bookmaker}")

            if not cle_odds_api:
                st.info(
                    "ℹ️ Value Bet Detector non configuré : ajoutez votre clé "
                    "[The-Odds-API](https://the-odds-api.com) dans `.streamlit/secrets.toml` "
                    "(`[odds_api]` puis `api_key = \"...\"`) pour comparer nos probabilités "
                    "aux cotes en direct."
                )
            else:
                if not cotes_match:
                    st.info(
                        "Cotes indisponibles pour ce match pour le moment "
                        "(marché pas encore ouvert, ou match non couvert par les bookmakers suivis)."
                    )
                else:
                    col_cote1, col_cote2 = st.columns(2)
                    with col_cote1:
                        st.metric(f"Cote {nom_notre_equipe}", f"{cotes_match['cote_nous']:.2f}")
                    with col_cote2:
                        st.metric(f"Cote {match_du_jour['adversaire']}", f"{cotes_match['cote_adverse']:.2f}")

                    for niveau, message in (
                        evaluer_value_bet(
                            pct_nous, cotes_match['cote_nous'], nom_notre_equipe, cotes_match['bookmaker']
                        ),
                        evaluer_value_bet(
                            pct_adverse, cotes_match['cote_adverse'], match_du_jour['adversaire'], cotes_match['bookmaker']
                        ),
                    ):
                        if not message:
                            continue
                        if niveau == 'value':
                            st.success(message)
                        elif niveau == 'evitez':
                            st.error(message)
                        else:
                            st.info(message)

                    st.caption(
                        f"Cotes Moneyline (marché h2h) fournies par {cotes_match['bookmaker']} "
                        "via The-Odds-API. Probabilité implicite = (1 / cote) × 100 ; "
                        "Value = notre probabilité algorithmique − probabilité implicite du marché."
                    )

            st.markdown("---")
            st.subheader("📊 Module de prédiction des Runs")

            # `prediction_runs` a déjà été calculé plus haut, avant la
            # "Recommandation de Pari Optimisée" (voir commentaire à cet endroit).
            if prediction_runs is None:
                st.info("Pas assez de données récentes pour estimer les runs de cette équipe.")
            else:
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

            # `joueurs_a_surveiller` a déjà été calculé plus haut, avant la
            # "Recommandation de Pari Optimisée" (voir commentaire à cet endroit).
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