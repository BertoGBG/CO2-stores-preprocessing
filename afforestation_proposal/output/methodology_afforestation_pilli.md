% Paste this into your Overleaf .tex file (inside \begin{document})
% Requires: amsmath, booktabs, siunitx, natbib or biblatex

\subsection{Afforestation as a Carbon Dioxide Removal Technology}
\label{sec:afforestation_methodology}

Afforestation---the establishment of forest on land not previously forested---is
modelled as a spatially resolved carbon dioxide removal (CDR) technology within
PyPSA-Eur. The implementation requires three inputs per network node: (i)~the
available land area from CORINE land cover, (ii)~a CO$_2$ sequestration rate
in \si{tCO_2\per ha\per yr}, and (iii)~cost parameters. This section describes
the derivation of the sequestration rate from European National Forest Inventory
(NFI) data.

\subsubsection{Data Source: Pilli et al.\ Yield Table Library}

The sequestration rates are derived from the JRC forest growth library compiled
by \citet{Pilli2024}, which provides age-class-resolved volume and increment
curves for 222~forest types across 25~EU Member States (EU-27 excluding Cyprus
and Malta). The library contains three key datasets:

\begin{itemize}
    \item \textbf{Standing stock} (m$^3$\,ha$^{-1}$): net merchantable volume
          under bark by 10-year age class for even-aged stands.
    \item \textbf{Net Annual Increment} (NAI, m$^3$\,ha$^{-1}$\,yr$^{-1}$):
          merchantable volume growth rate by age class, harmonised across
          countries using species-specific correction factors for bark,
          branches, top, and stump components.
    \item \textbf{Biomass Conversion and Expansion Factors} (BCEF,
          t$_\text{DM}$\,m$^{-3}$): age-class-resolved factors converting
          merchantable volume to total aboveground dry biomass, derived from
          species-specific allometric equations \citep{Boudewyn2007}.
\end{itemize}

The data are classified by country, administrative region (mostly NUTS-2),
forest type (leading species), management type, and management strategy
(even-aged or uneven-aged). The original NFI reference years span 1992--2018,
with most countries reporting data from the 2005--2015 period.

\subsubsection{Rotation-Averaged Sequestration Rate}

For new afforestation, we compute a \emph{rotation-averaged} Mean Annual
Increment (MAI), which represents the long-term average CO$_2$ uptake rate
over a full forest rotation. This approach is consistent with the methodology
used by the JRC Carbon Budget Model (EU-CBM-HAT) \citep{Pilli2024,
Grassi2018} and avoids the overestimation that would result from using
peak-growth-phase increment values.

For each combination of forest type~$f$, region~$r$, and management type~$m$,
the computation proceeds as follows:

\paragraph{Step 1: Determine optimal rotation age.}
The rotation age $T^*$ is defined as the age at which the volumetric MAI is
maximised:
\begin{equation}
    T^* = \arg\max_{T \geq T_\text{min}} \frac{V(T)}{T}
    \label{eq:rotation_age}
\end{equation}
where $V(T)$ is the standing stock (m$^3$\,ha$^{-1}$) at age~$T$ and
$T_\text{min} = \SI{25}{yr}$ is a minimum rotation age imposed to avoid
numerical artefacts at very young age classes. The age classes are reported
at 10-year intervals with midpoints at 5, 15, 25, \ldots, \SI{195}{yr}.

\paragraph{Step 2: Compute volumetric MAI.}
The rotation-averaged volumetric MAI is:
\begin{equation}
    \text{MAI}_\text{vol} = \frac{V(T^*)}{T^*}
    \quad [\text{m}^3\,\text{ha}^{-1}\,\text{yr}^{-1}]
    \label{eq:mai_vol}
\end{equation}

\paragraph{Step 3: Convert volume to CO$_2$.}
The volumetric MAI is converted to a CO$_2$ sequestration rate using the BCEF
at the rotation age, the IPCC carbon fraction, and a root-to-shoot expansion:
\begin{equation}
    \text{MAI}_{\text{CO}_2} = \text{MAI}_\text{vol}
        \times \text{BCEF}(T^*)
        \times C_f
        \times \frac{M_{\text{CO}_2}}{M_C}
        \times (1 + R)
    \quad [\text{tCO}_2\,\text{ha}^{-1}\,\text{yr}^{-1}]
    \label{eq:mai_co2}
\end{equation}
where:
\begin{itemize}
    \item $\text{BCEF}(T^*)$ is the Biomass Conversion and Expansion Factor
          at the rotation age (t$_\text{DM}$\,m$^{-3}$), converting
          merchantable volume to total aboveground biomass;
    \item $C_f = 0.5$ is the carbon fraction of dry biomass
          \citep{IPCC2006};
    \item $M_{\text{CO}_2}/M_C = 44/12 \approx 3.667$ is the molecular mass
          ratio of CO$_2$ to carbon;
    \item $R = 0.25$ is the root-to-shoot ratio, accounting for belowground
          biomass \citep{IPCC2006, Mokany2006}.
\end{itemize}

\paragraph{Step 4: Regional aggregation.}
For each NUTS-2 region, the sequestration rate is computed as the
equal-weight average across all productive forest types present in that
region:
\begin{equation}
    \overline{\text{MAI}}_{\text{CO}_2, r}
    = \frac{1}{|F_r|} \sum_{f \in F_r} \text{MAI}_{\text{CO}_2, f, r}
    \label{eq:nuts2_agg}
\end{equation}
where $F_r$ is the set of forest types with data in region~$r$.
Only even-aged, productive high-forest stands are included (management types
``H'', ``HP'', ``HS'', and ``MAN''); coppice, non-productive, and protected
forests are excluded.

For countries where the Pilli et al.\ data are reported at NUTS-0 level
(Bulgaria, Estonia, Greece, Croatia, Hungary, Ireland, Lithuania, Luxembourg,
Latvia, the Netherlands, Slovenia, and Slovakia), the national-average rate is
applied uniformly to all NUTS-2 regions within that country. For countries
using non-standard regional classifications (Germany, Finland, Italy, Poland),
a mapping to NUTS-2 codes is applied following the correspondence tables
provided in the dataset.

\subsubsection{Consistency with EU Policy Frameworks}

The rotation-averaged MAI is consistent with the accounting rules of the
EU Land Use, Land-Use Change and Forestry (LULUCF) Regulation
\citep{EU2018841, EU2023839}, which requires gross-net accounting of
emissions and removals on afforested land during a 20--30~year transition
period. The use of rotation-averaged (rather than peak-phase) rates avoids
overestimating the CDR potential and is compatible with the monitoring
requirements of the EU Carbon Removal Certification Framework (CRCF)
\citep{EU20243012}.

\subsubsection{Integration in PyPSA-Eur}

Afforestation is represented as a \texttt{Store} component on a dedicated
CO$_2$ bus at each network node, connected to the \texttt{co2 atmosphere}
bus via a \texttt{Link}. The maximum storage capacity (e\_nom\_max) is
determined by the available CORINE land area, the sequestration rate, and
a configurable maximum land-usage fraction. The capital cost is computed
from establishment and maintenance costs, annualised over a configurable
lifetime (default \SI{30}{yr}, consistent with the LULUCF transition period).

The sequestration rates derived from the Pilli et al.\ data replace the
previous country-level biomass density approach and provide sub-national
spatial resolution at NUTS-2 level across 23~EU countries.

\subsubsection{Results and Validation}

The computed rotation-averaged sequestration rates range from
\SIrange{2.1}{12.6}{tCO_2\per ha\per yr} across 114~NUTS-2 regions, with
a European median of \SI{5.2}{tCO_2\per ha\per yr} and mean of
\SI{5.5}{tCO_2\per ha\per yr}. These values are consistent with the
\SIrange{5}{15}{tCO_2\per ha\per yr} range reported for temperate European
afforestation in the literature \citep{Nabuurs2013, Avitabile2024}.

As a validation case, the five Danish NUTS-2 regions yield rates of
\SIrange{3.9}{5.8}{tCO_2\per ha\per yr}, with a national average of
approximately \SI{5.1}{tCO_2\per ha\per yr}. This is in reasonable
agreement with the \SI{6.5}{tCO_2\per ha\per yr} estimated by
\citet{Fernandes2026} for a 60\%~conifer / 40\%~broadleaf species mix;
the difference is attributable to the equal-weight species averaging used
here versus an area-weighted approach that gives greater weight to
higher-productivity conifer species.

Table~\ref{tab:affo_rates_sample} reports a selection of regional rates.

\begin{table}[htbp]
\centering
\caption{Rotation-averaged CO$_2$ sequestration rates for selected NUTS-2
regions. $\bar{T}$: mean rotation age; $n$: number of forest types.}
\label{tab:affo_rates_sample}
\sisetup{round-mode=places, round-precision=1}
\begin{tabular}{llS[round-precision=1]S[round-precision=0]S[round-precision=0]}
\toprule
{Country} & {NUTS-2} & {Rate [tCO$_2$\,ha$^{-1}$\,yr$^{-1}$]}
          & {$\bar{T}$ [yr]} & {$n$} \\
\midrule
Austria   & AT21  & 6.8   & 47  & 8  \\
Belgium   & BE20  & 6.9   & 34  & 7  \\
Czechia   & CZ01  & 5.2   & 48  & 4  \\
Denmark   & DK01  & 5.8   & 39  & 7  \\
Denmark   & DK05  & 3.9   & 39  & 7  \\
Finland   & FI1A  & 2.5   & 25  & 3  \\
France    & FRF1  & 5.4   & 27  & 6  \\
Germany   & DE2   & 7.2   & 34  & 8  \\
Ireland   & IE    & 7.1   & 33  & 6  \\
Italy     & ITH5  & 5.6   & 40  & 14 \\
Poland    & SL    & 5.9   & 31  & 8  \\
Romania   & RO21  & 6.0   & 28  & 9  \\
Sweden    & SE11  & 4.3   & 100 & 4  \\
Sweden    & SE33  & 2.1   & 33  & 4  \\
\bottomrule
\end{tabular}
\end{table}


% ── References (add to your .bib file) ──────────────────────────────────────
%
% @article{Pilli2024,
%   author  = {Pilli, Roberto and Blujdea, Viorel and Rougieux, Paul},
%   title   = {Forest aboveground biomass and volume increment library for
%              {EU-27} Member States},
%   year    = {2024},
%   journal = {Zenodo},
%   doi     = {10.5281/zenodo.10214062},
%   note    = {JRC harmonised National Forest Inventory data}
% }
%
% @article{Avitabile2024,
%   author  = {Avitabile, Valerio and others},
%   title   = {Harmonised statistics and maps of forest biomass and increment
%              in {Europe}},
%   journal = {Scientific Data},
%   year    = {2024},
%   volume  = {11},
%   pages   = {274},
%   doi     = {10.1038/s41597-024-03107-w}
% }
%
% @article{Nabuurs2013,
%   author  = {Nabuurs, Gert-Jan and Lindner, Marcus and Verkerk, Pieter J.
%              and others},
%   title   = {First signs of carbon sink saturation in {European} forest
%              biomass},
%   journal = {Nature Climate Change},
%   year    = {2013},
%   volume  = {3},
%   pages   = {792--796},
%   doi     = {10.1038/nclimate1853}
% }
%
% @article{Grassi2018,
%   author  = {Grassi, Giacomo and others},
%   title   = {Reconciling global-model estimates and country reporting of
%              anthropogenic forest {CO$_2$} sinks},
%   journal = {Nature Climate Change},
%   year    = {2018},
%   volume  = {8},
%   pages   = {914--920},
%   doi     = {10.1038/s41558-018-0283-x}
% }
%
% @incollection{IPCC2006,
%   author    = {{IPCC}},
%   title     = {Agriculture, Forestry and Other Land Use},
%   booktitle = {2006 {IPCC} Guidelines for National Greenhouse Gas
%                Inventories},
%   year      = {2006},
%   volume    = {4},
%   publisher = {IGES, Japan}
% }
%
% @article{Mokany2006,
%   author  = {Mokany, Karel and Raison, R. John and Prokushkin, Anatoly S.},
%   title   = {Critical analysis of root:shoot ratios in terrestrial biomes},
%   journal = {Global Change Biology},
%   year    = {2006},
%   volume  = {12},
%   pages   = {84--96},
%   doi     = {10.1111/j.1365-2486.2005.001043.x}
% }
%
% @techreport{Boudewyn2007,
%   author      = {Boudewyn, P. and Song, X. and Magnussen, S. and Gillis, M. D.},
%   title       = {Model-based, volume-to-biomass conversion for forested and
%                  vegetated land in {Canada}},
%   institution = {Natural Resources Canada, Canadian Forest Service},
%   year        = {2007},
%   number      = {BC-X-411}
% }
%
% @misc{EU2018841,
%   author = {{European Parliament and Council}},
%   title  = {Regulation ({EU}) 2018/841 on the inclusion of greenhouse gas
%             emissions and removals from land use, land use change and
%             forestry ({LULUCF})},
%   year   = {2018}
% }
%
% @misc{EU2023839,
%   author = {{European Parliament and Council}},
%   title  = {Regulation ({EU}) 2023/839 amending Regulation ({EU}) 2018/841
%             ({LULUCF} amendment)},
%   year   = {2023}
% }
%
% @misc{EU20243012,
%   author = {{European Parliament and Council}},
%   title  = {Regulation ({EU}) 2024/3012 establishing a {Union} certification
%             framework for carbon removals ({CRCF})},
%   year   = {2024}
% }
%
% @article{Fernandes2026,
%   author  = {Fernandes, Ana and Alamia, Alberto and others},
%   title   = {Afforestation as a negative emission technology in
%              {European} energy system optimisation},
%   journal = {[in preparation]},
%   year    = {2026}
% }
