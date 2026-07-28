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
from datetime import datetime   # Gestion des dates

import statsapi                 # Utilisation de l'API MLB

# Année courante MLB (toujours dynamique !)
ANNEE_COURANTE = datetime.now().year

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
    """Récupère la liste des équipes MLB pour l'année donnée (par défaut année courante)."""
    if year is None:
        year = ANNEE_COURANTE
    season_teams = statsapi.get('teams', {'sportIds': 1, 'season': year})
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
    for t in statsapi.get('teams', {'sportIds': 1, 'season': year})['teams']:
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
def get_scoreurs_runs_match(game_id: int, est_domicile: bool):
    """
    Récupère, via le boxscore statsapi d'un match, la liste des joueurs de l'équipe
    (domicile ou extérieur) ayant marqué au moins un run lors de ce match.
    Retourne une liste de dicts {'name': str, 'runs': int}.
    """
    if not game_id:
        return []
    try:
        box = statsapi.boxscore_data(int(game_id))
        batters = box.get('homeBatters', []) if est_domicile else box.get('awayBatters', [])

        runs_par_joueur = {}
        for b in batters:
            if not b.get('personId'):
                continue  # ligne d'en-tête du tableau, pas un joueur
            try:
                runs = int(b.get('r', 0) or 0)
            except (ValueError, TypeError):
                continue
            if runs > 0:
                nom = b.get('name', 'Inconnu')
                runs_par_joueur[nom] = runs_par_joueur.get(nom, 0) + runs

        return [{'name': nom, 'runs': runs} for nom, runs in runs_par_joueur.items()]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_matchs_avec_scoreurs(annee: int, equipe_abbr: str):
    """
    Enrichit les données de match avec la liste des scoreurs de runs par match,
    et calcule le cumul de runs marqués par joueur sur toute la période chargée.
    Retourne (df_matchs_enrichi, df_meilleurs_scoreurs).
    """
    df = charger_donnees_equipe(annee, equipe_abbr)
    if df.empty or 'game_id' not in df.columns:
        return df, pd.DataFrame()

    df = df.copy()
    colonne_joueurs = []
    cumul_runs = {}

    for _, ligne in df.iterrows():
        scoreurs = get_scoreurs_runs_match(ligne['game_id'], bool(ligne['Est_Domicile']))
        # Chaque cellule liste "Nom (runs)" par joueur, séparés par des virgules
        entrees = [f"{s['name']} ({s['runs']})" for s in scoreurs]
        colonne_joueurs.append(", ".join(entrees) if entrees else "—")
        for s in scoreurs:
            cumul_runs[s['name']] = cumul_runs.get(s['name'], 0) + s['runs']

    df['Joueurs (Runs)'] = colonne_joueurs

    df_meilleurs = pd.DataFrame(
        [{'Joueur': nom, 'Runs Marqués': total} for nom, total in cumul_runs.items()]
    )
    if not df_meilleurs.empty:
        df_meilleurs = df_meilleurs.sort_values('Runs Marqués', ascending=False).reset_index(drop=True)

    return df, df_meilleurs


def calculer_resume_10_derniers_matchs(df_derniers: pd.DataFrame):
    """
    À partir des données brutes des 10 derniers matchs (colonnes 'R' et 'Joueurs (Runs)'),
    calcule :
      - la moyenne de runs marqués sur ces matchs
      - le top 3 des joueurs les plus récurrents (somme des runs marqués), en gérant
        la séparation des noms lorsqu'ils sont regroupés dans la même cellule.
    Retourne (moyenne_runs, liste_top3) où liste_top3 est une liste de tuples (nom, total_runs).
    """
    if df_derniers.empty or 'R' not in df_derniers.columns:
        return None, []

    moyenne_runs = pd.to_numeric(df_derniers['R'], errors='coerce').mean()

    cumul_joueurs = {}
    for cellule in df_derniers.get('Joueurs (Runs)', []):
        if not cellule or cellule == "—":
            continue
        # Une cellule peut contenir plusieurs joueurs séparés par des virgules, ex:
        # "Freeman, F (2), Muncy (1), Hernández, T (1)". Certains noms MLB contiennent
        # eux-mêmes une virgule (format "Nom, Initiale"), donc on ne peut pas simplement
        # découper sur toutes les virgules : on découpe plutôt sur chaque entrée complète
        # "... (N)" (recherche non-gourmande jusqu'à la prochaine parenthèse de run).
        entrees = re.findall(r'(.+?\(\d+\))(?:,\s*|$)', cellule)
        for entree in entrees:
            entree = entree.strip()
            if not entree:
                continue
            correspondance = re.match(r'^(.*)\((\d+)\)$', entree)
            if correspondance:
                nom = correspondance.group(1).strip()
                runs = int(correspondance.group(2))
            else:
                nom = entree
                runs = 1
            cumul_joueurs[nom] = cumul_joueurs.get(nom, 0) + runs

    top_3 = sorted(cumul_joueurs.items(), key=lambda x: x[1], reverse=True)[:3]
    return moyenne_runs, top_3

# ============================================================
# 4. LISTE DES ÉQUIPES MLB (Abréviations officielles)
# ============================================================

# ============================================================
# 5. INTERFACE PRINCIPALE
# ============================================================

st.title("⚾ Analyse Statistiques MLB")
st.markdown("### Explorez les runs, les sluggers récurrents et les tendances W/L")

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
    - **HR** : Home Runs (Coup de circuit)
    - **SLG** : Slugging Percentage (Pourcentage de puissance)
    - **RBI** : Runs Batted In (Points produits)
    - **AVG** : Batting Average (Moyenne de frappe)
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
    "⚡ Sluggers Récurrents"
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

    # Chargement des données de matchs, enrichies avec les scoreurs de runs (boxscores statsapi)
    with st.spinner(f"Chargement des données et des boxscores pour les {EQUIPES_MLB[equipe_abbr]} ({annee})... (peut prendre un moment)"):
        df_matchs, df_meilleurs_scoreurs = get_matchs_avec_scoreurs(annee, equipe_abbr)

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

            roster_data = statsapi.get('team_roster', {'teamId': team_id, 'season': annee})

            for item in roster_data.get('roster', []):
                player_id = item['person']['id']
                nom = item['person']['fullName']

                player_stats = statsapi.player_stat_data(player_id, group="hitting", type="season")

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
        display_columns = ['Date', 'Équipe Domicile', 'Équipe Extérieur', 'R', 'RA', 'W/L', 'Joueurs (Runs)']
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
        moyenne_10_derniers, top3_joueurs_10_derniers = calculer_resume_10_derniers_matchs(
            df_matchs.tail(10)
        )
        if moyenne_10_derniers is not None:
            if top3_joueurs_10_derniers:
                texte_top3 = ", ".join(
                    f"{nom} ({runs} runs)" for nom, runs in top3_joueurs_10_derniers
                )
            else:
                texte_top3 = "Aucun joueur enregistré"

            st.markdown(f"**Moyenne de runs sur les 10 derniers matchs : {moyenne_10_derniers:.2f}**")
            st.markdown(f"**Top 3 des joueurs les plus récurrents (runs marqués) : {texte_top3}**")
        else:
            st.markdown("**Résumé indisponible : pas assez de données sur les 10 derniers matchs.**")

        st.markdown("---")
        st.subheader("🏅 Meilleurs scoreurs de Runs")
        st.markdown(f"Cumul des runs marqués par joueur sur la saison {annee} pour les {EQUIPES_MLB[equipe_abbr]}")
        if not df_meilleurs_scoreurs.empty:
            st.dataframe(
                df_meilleurs_scoreurs,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Aucune donnée de scoreurs disponible pour cette équipe/saison.")
    elif not df_matchs.empty:
        st.warning("Données de runs non disponibles pour cette équipe.")
    else:
        st.error("Impossible de charger les données. Vérifiez l'abréviation de l'équipe.")

# --------------------------------------------------------------
# ONGLET 2: SLUGGERS RÉCURRENTS
# --------------------------------------------------------------
with onglets[1]:
    st.header("⚡ Analyse des Sluggers Récurrents")
    st.markdown("Les frappeurs de puissance avec les meilleures statistiques de la ligue")

    # Chargement des statistiques de frappe par statsapi
    @st.cache_data
    def charger_statistiques_frappe_joueurs(annee: int = None) -> pd.DataFrame:
        """Récupère les stats des frappeurs MLB sur la saison demandée (défaut = année courante)."""
        if annee is None:
            annee = ANNEE_COURANTE
        try:
            equipes = get_teams_mlb_this_year(annee)
            team_ids = get_team_ids_dict(annee)
            all_batters = []
            for abbr, nom in equipes.items():
                tid = team_ids.get(abbr)
                if tid is None:
                    continue

                try:
                    stats = statsapi.team_year_batting(tid, season=annee, splits=False)
                    for joueur in stats:
                        nom_joueur = joueur.get('playerFullName', '')
                        if not nom_joueur:
                            continue
                        ab = joueur.get('atBats', 0)
                        try:
                            ab = int(ab)
                        except Exception:
                            ab = 0
                        try:
                            record = {
                                "Name": nom_joueur,
                                "Team": abbr,
                                "HR": int(joueur.get('homeRuns', 0) or 0),
                                "AVG": float(joueur.get('avg', 0) or 0),
                                "SLG": float(joueur.get('slg', 0) or 0),
                                "RBI": int(joueur.get('rbi', 0) or 0),
                                "AB": ab,
                                "H": int(joueur.get('hits', 0) or 0)
                            }
                        except Exception:
                            continue
                        all_batters.append(record)
                except Exception:
                    continue

            df = pd.DataFrame(all_batters)
            if df.empty:
                return pd.DataFrame()
            for field in ['HR', 'RBI', 'AB', 'H']:
                df[field] = pd.to_numeric(df[field], errors='coerce')
            for field in ['AVG', 'SLG']:
                df[field] = pd.to_numeric(df[field], errors='coerce')
            df = df.dropna(subset=['Name', 'AB', 'HR', 'SLG'])
            df = df[df['AB'] > 0]

            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des statistiques de frappe : {e}")
            return pd.DataFrame()

    with st.spinner(f"Chargement des statistiques de frappe pour la saison {annee}..."):
        df_stats = charger_statistiques_frappe_joueurs(annee)

    if not df_stats.empty:
        if 'AB' in df_stats.columns:
            df_stats = df_stats[df_stats['AB'] >= 100]

        sous_onglets = st.tabs(["🏆 Top Home Runs", "💪 Top Slugging", "📊 Classement Complet"])

        # Sous-onglet 1: Top Home Runs
        with sous_onglets[0]:
            st.subheader("🏆 Top 15 des Frappeurs de Circuits")
            if 'HR' in df_stats.columns:
                top_hr = df_stats.nlargest(15, 'HR')[['Name', 'Team', 'HR', 'RBI', 'AVG', 'SLG']]
                import plotly.express as px
                fig_hr = px.bar(
                    top_hr.head(10),
                    x='Name',
                    y='HR',
                    title=f"Top 10 Frappeurs de Circuits - Saison {annee}",
                    color='HR',
                    color_continuous_scale='Reds',
                    labels={'Name': 'Joueur', 'HR': 'Home Runs'}
                )

                fig_hr.update_layout(
                    template="plotly_white",
                    height=400,
                    xaxis_title="Joueur",
                    yaxis_title="Nombre de Home Runs"
                )
                fig_hr.update_xaxes(tickangle=45)
                st.plotly_chart(fig_hr, use_container_width=True)
                st.markdown("---")
                st.dataframe(top_hr, use_container_width=True, hide_index=True)

        # Sous-onglet 2: Top Slugging
        with sous_onglets[1]:
            st.subheader("💪 Top 15 des Frappeurs de Puissance (Slugging)")
            if 'SLG' in df_stats.columns:
                top_slg = df_stats.nlargest(15, 'SLG')[['Name', 'Team', 'SLG', 'HR', 'AVG', 'AB']]
                top_slg_display = top_slg.copy()
                top_slg_display['SLG_%'] = (top_slg_display['SLG'] * 100).round(1)
                import plotly.express as px
                fig_slg = px.bar(
                    top_slg.head(10),
                    x='Name',
                    y='SLG',
                    title=f"Top 10 Slugging Percentage - Saison {annee}",
                    color='SLG',
                    color_continuous_scale='Blues',
                    labels={'Name': 'Joueur', 'SLG': 'Slugging %'}
                )

                fig_slg.update_layout(
                    template="plotly_white",
                    height=400,
                    xaxis_title="Joueur",
                    yaxis_title="Slugging Percentage"
                )
                fig_slg.update_xaxes(tickangle=45)
                st.plotly_chart(fig_slg, use_container_width=True)
                st.markdown("---")
                st.dataframe(top_slg_display[['Name', 'Team', 'SLG_%', 'HR', 'AVG']].rename(
                    columns={'SLG_%': 'SLG (%)', 'AVG': 'Avg'}
                ), use_container_width=True, hide_index=True)

        # Sous-onglet 3: Classement Complet
        with sous_onglets[2]:
            st.subheader("📊 Classement Complet des Frappeurs")

            df_classement = df_stats[['Name', 'Team', 'HR', 'SLG', 'RBI', 'AVG', 'AB']].copy()
            if 'SLG' in df_classement.columns:
                df_classement['SLG_%'] = (df_classement['SLG'] * 100).round(1)
            if 'AVG' in df_classement.columns:
                df_classement['AVG'] = df_classement['AVG'].round(3)
            df_classement = df_classement.rename(columns={
                'SLG_%': 'SLG (%)',
                'AB': 'Passages'
            })

            st.dataframe(
                df_classement.sort_values('HR', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            csv = df_classement.sort_values('HR', ascending=False).to_csv(index=False)
            st.download_button(
                label="📥 Télécharger les données (CSV)",
                data=csv,
                file_name=f"stats_frappeurs_mlb_{annee}.csv",
                mime="text/csv"
            )
    else:
        st.error("Impossible de charger les statistiques de frappe.")

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