%% LaTeX2e class for student theses
%% sdthesis.cls
%%
%% Karlsruhe Institute of Technology
%% Institute for Program Structures and Data Organization
%% Chair for Software Design and Quality (SDQ)
%%
%% Dr.-Ing. Erik Burger
%% burger@kit.edu
%%
%% See https://sdq.kastel.kit.edu/wiki/Dokumentvorlagen
%%
%% Version 1.5, 2024-02-12

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdqthesis}[2024-02-12 v1.5 Student thesis class]
% Language options. German is also needed in English thesis for the abstract
\DeclareOption{ngerman}{\PassOptionsToPackage{\CurrentOption}{babel}}
\DeclareOption{english}{\PassOptionsToPackage{main=\CurrentOption,ngerman}{babel}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax
% Used to detect language families
\RequirePackage{translations}

%% ---------------------------------------------------------------
%% | Based on KOMAscript by Markus Kohm http://www.komascript.de |
%% ---------------------------------------------------------------
\LoadClass{scrbook}

%% --------------
%% | Typography |
%% --------------

% T1 font encoding
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

% serif type: Linux Libertine
\RequirePackage{libertine}
% Linux Libertine in math mode
\RequirePackage[libertine]{newtxmath}

% grotesque type: Adobe Source Sans Pro
\RequirePackage[scaled=.92]{sourcesanspro}

% monospace type: Bera Mono
\RequirePackage[scaled=.78]{beramono}

% semi-bold type (for subsections and paragraphs)
\newcommand*{\sbdefault}{sb}
\DeclareRobustCommand{\sbseries}{%
  \not@math@alphabet\sbseries\relax
  \fontseries\sbdefault\selectfont}

\DeclareTextFontCommand{\textsb}{\sbseries}

\addtokomafont{subsection}{\sbseries}
\addtokomafont{subsubsection}{\sbseries}
\addtokomafont{paragraph}{\sbseries}

% microtype for nicer typography
\RequirePackage[protrusion=true,expansion=true]{microtype}

% commands for code and model keywords in text
\newcommand{\code}[1]{\texttt{\hyphenchar\font45\relax #1}}
\newcommand{\model}[1]{\textsf{#1}}

% nicer spacing for enumerations
\RequirePackage{enumitem}

%% -----------
%% | Spacing |
%% -----------

% more flexible spacing for paragraphs
\RedeclareSectionCommands[
    beforeskip= .7em plus .6em minus .3em
]{paragraph}

%% ---------------
%% | Page Layout |
%% ---------------
\KOMAoptions{
   fontsize=12pt,
   paper=a4,
   parskip=half,
   titlepage=true,
   headinclude=true,
   footinclude=false,
   headsepline,
   headlines=1.1,
   bibliography=totoc,
}

% Margins
\typearea[20mm]{16} % twoside

\raggedbottom

\recalctypearea

%% Caption sizes
\setkomafont{caption}{\small}
\setkomafont{captionlabel}{\sffamily\bfseries\small}
\setcapindent{0em}

\clubpenalty=10000 %prevent orphans
\widowpenalty=10000 %prevent widows
\displaywidowpenalty10000

%% allow fuller pages
\renewcommand{\topfraction}{0.9}	% max fraction of floats at top
\renewcommand{\bottomfraction}{0.8}	% max fraction of floats at bottom
\renewcommand{\textfraction}{0.07}	% allow minimal text w. figs
% Parameters for FLOAT pages (not text pages):
\renewcommand{\floatpagefraction}{0.85}	% require fuller float pages
% N.B.: floatpagefraction MUST be less than topfraction !!
\renewcommand{\dblfloatpagefraction}{0.85}	% require fuller float pages


%% --------------
%% | Title page |
%% --------------

% Logos
\newcommand{\thegrouplogo}{sdqlogo}
\newcommand{\grouplogo}[1]{\renewcommand{\thegrouplogo}{#1}}
\newcommand{\printgrouplogo}{\includegraphics[height=1.8cm]{logos/\thegrouplogo}}
\newcommand{\nogrouplogo}{\renewcommand{\printgrouplogo}{}}

\titlehead{%
% KIT logo german/english
\ifcurrentbaselanguage{English}
{\includegraphics[height=1.8cm]{logos/kitlogo_en_cmyk}}
{\includegraphics[height=1.8cm]{logos/kitlogo_de_cmyk}}
\hfill
% SDQ logo
\printgrouplogo
\ThisCenterWallPaper{1}{title-background.pdf}
}

%
\setkomafont{title}{\huge\sffamily\bfseries}
\setkomafont{subtitle}{\normalfont\large}

\uppertitleback{Karlsruher Institut für Technologie\\ Fakultät für Informatik\\ Postfach 6980\\ 76128 Karlsruhe}

%% variables for title page
\newcommand{\theinstitute}{\ifcurrentbaselanguage{English}
    {KASTEL -- Institute of Information Security and Dependability}
    {KASTEL -- Institut für Informationssicherheit und Verlässlichkeit}}
\newcommand{\thethesistype}{}
\newcommand{\thereviewerone}{}
\newcommand{\thereviewertwo}{}
\newcommand{\theadvisorone}{}
\newcommand{\theadvisortwo}{}
\newcommand{\theeditstart}{}
\newcommand{\theeditend}{}

%% formatting commands for titlepage
\newcommand{\thesistype}[1]{\subtitle{\vskip2em #1 \ifcurrentbaselanguage{English}{of}{von}}%
\renewcommand{\thethesistype}{#1}}
\newcommand{\myinstitute}[1]{\renewcommand{\theinstitute}{#1}}
\newcommand{\reviewerone}[1]{\renewcommand{\thereviewerone}{#1}}
\newcommand{\reviewertwo}[1]{\renewcommand{\thereviewertwo}{#1}}
\newcommand{\advisorone}[1]{\renewcommand{\theadvisorone}{#1}}
\newcommand{\advisortwo}[1]{\renewcommand{\theadvisortwo}{#1}}

\newcommand{\editingtime}[2]{%
\renewcommand{\theeditstart}{#1}%
\renewcommand{\theeditend}{#2}%
%% do not show the date
\date{}
}

\newcommand{\settitle}{%
\publishers{%
\large
\ifcurrentbaselanguage{English}{At the KIT Department of Informatics}%
{An der KIT-Fakultät für Informatik}\\
\theinstitute\\[2em]
\begin{tabular}{l l}
  \ifcurrentbaselanguage{English}{First examiner}{1. Prüfer/Prüferin}: & \thereviewerone\\
  \ifcurrentbaselanguage{English}{Second examiner}{2. Prüfer/Prüferin}: & \thereviewertwo\\[.5em]
  \ifcurrentbaselanguage{English}{First advisor}{1. Betreuer/Betreuerin}: &  \theadvisorone\\
  % if there is no second advisor, do not output this line
  \ifthenelse{\equal{\theadvisortwo}{}}{}{%
  \ifcurrentbaselanguage{English}{Second advisor}{2. Betreuer/Betreuerin}: & \theadvisortwo\\
  }
\end{tabular}
\vskip2em
\theeditstart{} -- \theeditend
}
}

%% -----------------------------
%% | Abstract/Acknowledgements |
%% -----------------------------

\newcommand{\abstract}[1][\abstractname]{\chapter*{#1}}
\newcommand{\Abstract}[1][\abstractname]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}}

\def\ackname{Acknowledgments}
\def\abstractname{Abstract}
\def\switcht@deutsch{\svlanginfo
	\def\ackname{Danksagung}
	\def\abstractname{Kurzfassung}
}
\def\switcht@english{\svlanginfo
	\def\ackname{Acknowledgements}
	\def\abstractname{Abstract}
}

%% In English theses, an additional German Abstract is needed.
\newcommand{\includeabstract}{
\ifcurrentbaselanguage{English}{
% include English and German abstracts
\input{sections/abstract_en.tex}
\begin{otherlanguage}{ngerman}
\input{sections/abstract_de.tex}
\end{otherlanguage}
}{
% include only German abstract
\include{sections/abstract_de}
}
}

%% ------------
%% | Packages |
%% ------------

% draft mode
\RequirePackage{ifdraft}
\RequirePackage{ifthen}

% enumerate subsubsections
\setcounter{secnumdepth}{3}

% wallpaper for title page (KIT-Frame)
\RequirePackage{wallpaper}

% subimport of files
\RequirePackage{import}

% languages
\RequirePackage{babel}
\RequirePackage{csquotes}

% hyphenation for compound words
\RequirePackage[shortcuts]{extdash}

% nice tables
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{array}

% show graphics in draft mode
\RequirePackage{graphicx}
\setkeys{Gin}{draft=false}

% appendix
\RequirePackage[toc,title,header]{appendix}
\noappendicestocpagenum



% To enhance the funcationlity of latex tables
\RequirePackage{tabularx}
% Simplifies creating table cells that span multiple lines
\RequirePackage{makecell}

% For displaying algorithm pseudocode
\RequirePackage{algorithm}
\RequirePackage[noend]{algpseudocode}
% Based on https://texdoc.org/serve/algorithms/0
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
\newcommand{\algorithmautorefname}{Algorithm}
\algnewcommand{\PersistentState}{\item[\textbf{Persistent State:}]}

% Allows syntax highlighting
\RequirePackage{listings}
\RequirePackage{xcolor}

% Provides additional text symbols
\RequirePackage{textcomp}

% Allows table cells to span multiple rows.
\RequirePackage{multirow}

% Allows to rotate tables
\RequirePackage{rotating}

% \RequirePackage{amsthm}
% \theoremstyle{definition}
% \newtheorem{definition}{Definition}
% \providecommand{\definitionautorefname}{Definition}

\RequirePackage{tcolorbox}
\newtcolorbox{parameterbox}[1][]{
    colback=gray!10,        
    colframe=gray!50,      
    coltitle=black,         
    fonttitle=\bfseries\Large,
    title=Parameter: #1,
}

\lstdefinestyle{customcode}{
    backgroundcolor=\color{gray!10},
    basicstyle=\ttfamily\small,
    breaklines=true,
    captionpos=b,
    keywordstyle=\color{blue},
    stringstyle=\color{red!60!black},
    numberstyle=\small\color{gray},
    numbers=left,
    numbersep=10pt,
    frame=single,
    framesep=5pt,
    rulecolor=\color{black!30},
    xleftmargin=15pt,
    emphstyle=\color{purple},
    emph={class,def,return,if,elif,else,from,import,for,in},
    morekeywords={self},
    keywordstyle=\color{blue},
    identifierstyle=\color{black},
    commentstyle=\color{green!50!black}\itshape,  % Removed \bfseries
    morecomment=[l]{\#},  % Single-line comments
    morecomment=[s]{'''}{'''},  % Multi-line comments with single quotes
    morecomment=[s]{"""}{"""},  % Multi-line comments with double quotes
}

\RequirePackage[hyphens]{url}
\RequirePackage[breaklinks,colorlinks=false]{hyperref}

\RequirePackage[acronym,nonumberlist,toc]{glossaries}
\makeglossaries
\setacronymstyle{long-short}

% PDF specific packages
\newcommand\setpdf{
    \hypersetup{%
    pdftitle={\thethesistype},%
    pdfsubject={\@title},%
    pdfauthor={\@author},%
    pdfborder={0 0 0},%
    }%
    \let\theauthor\@author
}