%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Class file for the abstract of the Bicycle and Motercycle Dynamics 
%% Symposium 2010, 20-22 October 2010, Delft, Th Netherlands 
%% JPM 20100221
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bmd2010a}[2010/02/21 BMD2010 abstract class]  
\ProcessOptions\relax

\DeclareOption{10pt}{\OptionNotUsed}  
\DeclareOption{12pt}{\OptionNotUsed}  
\DeclareOption{twocolumn}{\OptionNotUsed}  
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}  
  
\ProcessOptions\relax  
  
\LoadClass[11pt,a4paper,twoside]{article}  
  
%\RequirePackage[latin1]{inputenc}   % for unix users
\RequirePackage[ansinew]{inputenc}  %  for windows users
%\RequirePackage[aplemac]{inputenc}  % for mac users

\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{graphicx}
\RequirePackage{verbatim}
% \RequirePackage{graphicx}
\RequirePackage{tabularx}
\RequirePackage{amsmath,amsfonts,amssymb,amsthm,amsbsy,amsmath}
\RequirePackage{latexsym}
\RequirePackage{bm}
\RequirePackage{a4}
% \RequirePackage{fancyhdr}
\RequirePackage{engord}
\RequirePackage{epsfig,vmargin,booktabs}


\setpapersize{A4}  % needs vmargin package
%\setmarginsrb{left}{top}{right}{bottom}{headh1gt}{headsep}{foothgt}{footskip}
%\setmarginsrb{30mm}{10mm}{30mm}{10mm}{15mm}{10mm}{0mm}{0mm}
\setmarginsrb{30mm}{10mm}{30mm}{27mm}{15mm}{5mm}{0mm}{0mm}
% headsep:  Distance from bottom of header to the body of text on a page.
% footskip: Distance from bottom of body to the bottom of the footer (the footer is the text at the bottom of each page).


\parindent=0pt
%\parskip0pc
\parskip=2ex


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The LABEL extensions for proofing. J. Kaipio, 4.4.1995
%% CAPTIONS FOR TABLES AND FIGURES
%%
%%  There's a new logical variable \tablecap, which is
%%  set \true by saying \tablecaptrue and \false by
%%  \tablecapfalse. The variable is expanded by saying
%%  \iftablecap. Then there are two separate \caption
%%  macros: \tabcaption and \figcaption, the definition
%%  of which are obvious. The \iftablecap is performed in
%%  \@makecaption, which now contains two different
%%  captioning styles.
%%                               J. Kaipio 3.10.1990.
\newif\iftablecap
\def\tabcaption#1{\tablecaptrue\caption{#1}}
\def\figcaption#1{\tablecapfalse\caption{#1}}
\newlength{\t@bindwidth} \setlength{\t@bindwidth}{1cm}
\newlength{\c@pwidth} \setlength{\c@pwidth}{\hsize}
\addtolength{\c@pwidth}{-2\t@bindwidth}

\long\def\@makecaption#1#2{
   %\vskip 10pt
   \vskip 0.5ex
   \iftablecap                           % Is it a table?
     \setbox\@tempboxa\hbox{#2}
     \ifdim \wd\@tempboxa > \c@pwidth      % IF longer than one line-2cm:
         \hbox to\hsize{\hfil{\sc #1}\hfil}
         \begin{list}{}{\leftmargin\t@bindwidth
             \rightmargin\t@bindwidth}
             {\item \footnotesize  #2}
         \end{list}  
     \else                               % ELSE  center.
         \hbox to\hsize{\hfil{\sc #1}\hfil}
         \hbox to\hsize{\hfil{\footnotesize #2}\hfil}
     \fi                                   % of \ifdim
   \else                                 % Not a table: a picture
     \setbox\@tempboxa\hbox{\small{\bf #1}. #2}
     \ifdim \wd\@tempboxa >\c@pwidth       % IF longer than one line-2cm:
        \begin{list}{}{\leftmargin\t@bindwidth
           \rightmargin\t@bindwidth}
           {\small \item {\bf #1}. #2}
         \end{list}                        % THEN set as ordinary paragraph.
     \else                               % ELSE  center.
         \hbox to\hsize{\hfil\box\@tempboxa\hfil}
     \fi                                   % of \ifdim
   \fi                                   % of \if\tablecaptrue
   }                                     % \@makecaption

%  End of \@makecaption modifications.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SECTIONS 
\renewcommand\section{\@startsection {section}{1}{\z@}{-3.5ex plus -1ex minus-.2ex}{0.2ex}{\normalsize\bf}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% My defs....
% \newcommand{\dd}[1]{\mbox{${\mathrm d}{#1}$}} % d
% \newcommand{\grad}{\operatorname{grad}}
% \newcommand{\dev}{\operatorname{dev}}
% \newcommand{\tr}{\operatorname{tr}}
% \newcommand{\ov}[1]{\mbox{$\bm{#1}$}}                           % vectors
% \newcommand{\om}[1]{\mbox{$\bm{#1}$}}                           % matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\ps@plain}{%
  \renewcommand{\@oddhead}{{\parbox[b]{60ex}{\scriptsize%
    Bicycle and Motorcycle Dynamics 2010\\%
    Symposium on the Dynamics and Control of Single Track Vehicles,\\%
    20--22 October 2010, Delft, The Netherlands \vspace{5mm}}}}%
  \renewcommand{\@evenhead}{{\hfill\parbox[b]{60ex}{\scriptsize%
    \raggedleft Bicycle and Motorcycle Dynamics 2010\\%
    Symposium on the Dynamics and Control of Single Track Vehicles,\\%
    20--22 October 2010, Delft, The Netherlands \vspace{5mm}}}}%
  \renewcommand{\@oddfoot}{}%
  \renewcommand{\@evenfoot}{}%
}
\renewcommand{\baselinestretch}{0.91}

\AtBeginDocument{%    
  \pagestyle{plain}
}  
  
\endinput
