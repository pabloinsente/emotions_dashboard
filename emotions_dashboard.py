import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
alt.renderers.enable('svg')
alt.themes.enable('vox')
st.set_page_config(layout="wide")

import base64
import textwrap


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

svg_str = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" class="marks" width="919" height="349" viewBox="0 0 919 349"><rect width="919" height="349" fill="white"/><g fill="none" stroke-miterlimit="10" transform="translate(49,11)"><g class="mark-group role-frame root" role="graphics-object" aria-roledescription="group mark container"><g transform="translate(0,0)"><path class="background" aria-hidden="true" d="M0,0h0v300h0Z"/><g><g class="mark-group role-scope concat_0_group" role="graphics-object" aria-roledescription="group mark container"><g transform="translate(0,0)"><path class="background" aria-hidden="true" d="M0.5,0.5h400v300h-400Z" stroke="#ddd"/><g><g class="mark-group role-axis" aria-hidden="true"><g transform="translate(0.5,300.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-grid" pointer-events="none"><line transform="translate(0,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(40,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(80,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(120,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(160,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(200,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(240,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(280,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(320,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(360,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(400,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" aria-hidden="true"><g transform="translate(0.5,0.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-grid" pointer-events="none"><line transform="translate(0,300)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,270)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,240)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,210)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,180)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,150)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,120)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,90)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,60)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,30)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,0)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" role="graphics-symbol" aria-roledescription="axis" aria-label="X-axis titled \'Number of clusters - PCA\' for a linear scale with values from 0 to 100"><g transform="translate(0.5,300.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-tick" pointer-events="none"><line transform="translate(0,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(40,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(80,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(120,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(160,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(200,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(240,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(280,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(320,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(360,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(400,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-label" pointer-events="none"><text text-anchor="start" transform="translate(0,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0</text><text text-anchor="middle" transform="translate(40,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">10</text><text text-anchor="middle" transform="translate(80,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">20</text><text text-anchor="middle" transform="translate(120,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">30</text><text text-anchor="middle" transform="translate(160,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">40</text><text text-anchor="middle" transform="translate(200,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">50</text><text text-anchor="middle" transform="translate(240,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">60</text><text text-anchor="middle" transform="translate(280,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">70</text><text text-anchor="middle" transform="translate(320,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">80</text><text text-anchor="middle" transform="translate(360,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">90</text><text text-anchor="end" transform="translate(400,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">100</text></g><g class="mark-rule role-axis-domain" pointer-events="none"><line transform="translate(0,0)" x2="400" y2="0" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-title" pointer-events="none"><text text-anchor="middle" transform="translate(200,30)" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#000" opacity="1">Number of clusters - PCA</text></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" role="graphics-symbol" aria-roledescription="axis" aria-label="Y-axis titled \'Silhouette coefficients\' for a linear scale with values from 0.0 to 0.5"><g transform="translate(0.5,0.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-tick" pointer-events="none"><line transform="translate(0,300)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,270)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,240)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,210)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,180)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,150)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,120)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,90)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,60)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,30)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,0)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-label" pointer-events="none"><text text-anchor="end" transform="translate(-7,303)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.00</text><text text-anchor="end" transform="translate(-7,273)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.05</text><text text-anchor="end" transform="translate(-7,243)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.10</text><text text-anchor="end" transform="translate(-7,213)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.15</text><text text-anchor="end" transform="translate(-7,183)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.20</text><text text-anchor="end" transform="translate(-7,153)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.25</text><text text-anchor="end" transform="translate(-7,123)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.30</text><text text-anchor="end" transform="translate(-7,93.00000000000001)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.35</text><text text-anchor="end" transform="translate(-7,62.999999999999986)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.40</text><text text-anchor="end" transform="translate(-7,32.99999999999999)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.45</text><text text-anchor="end" transform="translate(-7,3)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0.50</text></g><g class="mark-rule role-axis-domain" pointer-events="none"><line transform="translate(0,300)" x2="0" y2="-300" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-title" pointer-events="none"><text text-anchor="middle" transform="translate(-32,150) rotate(-90) translate(0,-2)" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#000" opacity="1">Silhouette coefficients</text></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-line role-mark concat_0_marks" role="graphics-object" aria-roledescription="line mark container"><path aria-label="Number of clusters - PCA: 2; Silhouette coefficients: 0.322537025548" role="graphics-symbol" aria-roledescription="line mark" d="M8,106.47778467149176L12,67.11421652017381L16,11.8384885303783L20,53.39672586315434L24,69.3261290240064L28.000000000000004,64.05177904144679L32,65.08935743979605L36,84.88524297305302L40,88.97422477601663L44,91.62660441444284L48,84.04682780152851L52,89.53016610968147L56.00000000000001,98.18870885819334L60,93.42628195182726L64,95.42659704110045L68,105.40939820766638L72,118.83893231533985L76,118.37106821231792L80,103.25314130937714L84,112.94399285471168L88,117.33180422692892L92,117.32623108783787L96,113.68164321826276L100,125.27777247704451L104,124.53140813933997L108,122.72284877821143L112.00000000000001,114.23364885855385L115.99999999999999,120.39670003901166L120,118.4547192704582L124,114.31086920357711L128,121.84411184748667L132,118.5861403650317L136,122.26663830367337L140,113.75066388328408L144,115.40944939228059L148,114.327587236553L152,113.84659928907558L156,110.03363068834933L160,104.68643742813059L164,104.40889182999649L168,114.86170565665655L172,110.77826582970609L176,108.848316896987L180,100.60799145174876L184,116.58359549700059L188,102.10159862233822L192,109.74001911328199L196,108.54957817202316L200,103.56326653715202L204,101.9066521994749L208,113.18404561971968L212,114.03458388546227L216,115.2026765284517L220.00000000000003,107.85985607756675L224.00000000000003,115.58299031706039L227.99999999999997,110.66236285133137L231.99999999999997,104.6243852655805L236,100.06256905127891L240,108.79581761821674L244,110.5811907709109L248,104.6266597674309L252,106.33210145085594L256,105.82545431081707L260,116.36416585152764L264,116.27135427850752L268,98.7741588617841L272,102.08844220616992L276,101.59966936971068L280,105.93510793019247L284,111.50520642638352L288,103.51347668175255L292,107.92320567317026L296,108.28359439419344L300,99.95158508045974L304,104.11223549398295L308,105.9701612726746L312,103.89118845332183L316,111.12912338492768L320,115.96409643069188L324,114.60572883515779L328,110.80463981022201L332,107.21965122833006L336,111.82779244401783L340,111.05809691814255L344,108.82965680438947L348,111.1743008312248L352,101.3958593353092L356,117.95739033927416L360,118.94965924121334L364,112.98634411234826L368,108.96802522186411L372,110.671070172605L376,122.30886866851516L380,115.60772688596546L384,126.86285607865183L388,116.52695439947385L392,119.92229592801085L396,125.3465804618746" stroke="#4c78a8" stroke-width="2"/></g></g><path class="foreground" aria-hidden="true" d="" display="none"/></g></g><g class="mark-group role-scope concat_1_group" role="graphics-object" aria-roledescription="group mark container"><g transform="translate(463,0)"><path class="background" aria-hidden="true" d="M0.5,0.5h400v300h-400Z" stroke="#ddd"/><g><g class="mark-group role-axis" aria-hidden="true"><g transform="translate(0.5,300.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-grid" pointer-events="none"><line transform="translate(0,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(40,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(80,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(120,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(160,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(200,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(240,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(280,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(320,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(360,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(400,-300)" x2="0" y2="300" stroke="#ddd" stroke-width="1" opacity="1"/></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" aria-hidden="true"><g transform="translate(0.5,0.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-grid" pointer-events="none"><line transform="translate(0,300)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,275)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,250)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,225)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,200)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,175)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,150)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,125)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,100)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,75)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,50)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,25)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/><line transform="translate(0,0)" x2="400" y2="0" stroke="#ddd" stroke-width="1" opacity="1"/></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" role="graphics-symbol" aria-roledescription="axis" aria-label="X-axis titled \'Number of clusters - PCA\' for a linear scale with values from 0 to 100"><g transform="translate(0.5,300.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-tick" pointer-events="none"><line transform="translate(0,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(40,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(80,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(120,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(160,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(200,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(240,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(280,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(320,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(360,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(400,0)" x2="0" y2="5" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-label" pointer-events="none"><text text-anchor="start" transform="translate(0,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0</text><text text-anchor="middle" transform="translate(40,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">10</text><text text-anchor="middle" transform="translate(80,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">20</text><text text-anchor="middle" transform="translate(120,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">30</text><text text-anchor="middle" transform="translate(160,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">40</text><text text-anchor="middle" transform="translate(200,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">50</text><text text-anchor="middle" transform="translate(240,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">60</text><text text-anchor="middle" transform="translate(280,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">70</text><text text-anchor="middle" transform="translate(320,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">80</text><text text-anchor="middle" transform="translate(360,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">90</text><text text-anchor="end" transform="translate(400,15)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">100</text></g><g class="mark-rule role-axis-domain" pointer-events="none"><line transform="translate(0,0)" x2="400" y2="0" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-title" pointer-events="none"><text text-anchor="middle" transform="translate(200,30)" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#000" opacity="1">Number of clusters - PCA</text></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-group role-axis" role="graphics-symbol" aria-roledescription="axis" aria-label="Y-axis titled \'SSE\' for a linear scale with values from 0 to 240"><g transform="translate(0.5,0.5)"><path class="background" aria-hidden="true" d="M0,0h0v0h0Z" pointer-events="none"/><g><g class="mark-rule role-axis-tick" pointer-events="none"><line transform="translate(0,300)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,275)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,250)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,225)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,200)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,175)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,150)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,125)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,100)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,75)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,50)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,25)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/><line transform="translate(0,0)" x2="-5" y2="0" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-label" pointer-events="none"><text text-anchor="end" transform="translate(-7,303)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">0</text><text text-anchor="end" transform="translate(-7,278)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">20</text><text text-anchor="end" transform="translate(-7,253)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">40</text><text text-anchor="end" transform="translate(-7,228)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">60</text><text text-anchor="end" transform="translate(-7,203.00000000000003)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">80</text><text text-anchor="end" transform="translate(-7,177.99999999999997)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">100</text><text text-anchor="end" transform="translate(-7,153)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">120</text><text text-anchor="end" transform="translate(-7,127.99999999999999)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">140</text><text text-anchor="end" transform="translate(-7,103.00000000000001)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">160</text><text text-anchor="end" transform="translate(-7,78)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">180</text><text text-anchor="end" transform="translate(-7,52.999999999999986)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">200</text><text text-anchor="end" transform="translate(-7,28.00000000000001)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">220</text><text text-anchor="end" transform="translate(-7,3)" font-family="sans-serif" font-size="10px" fill="#000" opacity="1">240</text></g><g class="mark-rule role-axis-domain" pointer-events="none"><line transform="translate(0,300)" x2="0" y2="-300" stroke="#888" stroke-width="1" opacity="1"/></g><g class="mark-text role-axis-title" pointer-events="none"><text text-anchor="middle" transform="translate(-29,150) rotate(-90) translate(0,-2)" font-family="sans-serif" font-size="11px" font-weight="bold" fill="#000" opacity="1">SSE</text></g></g><path class="foreground" aria-hidden="true" d="" pointer-events="none" display="none"/></g></g><g class="mark-line role-mark concat_1_marks" role="graphics-object" aria-roledescription="line mark container"><path aria-label="Number of clusters - PCA: 1; SSE: 239.337849429" role="graphics-symbol" aria-roledescription="line mark" d="M4,0.8276882132059127L8,86.1156216908906L12,164.91918039004526L16,224.4165204247843L20,236.19910426652606L24,244.09447430469586L28.000000000000004,251.06640033252L32,256.4417693035304L36,261.0813984195204L40,264.5537113661176L44,267.39931286866647L48,269.7067071650422L52,271.5869221687457L56.00000000000001,273.24556415768615L60,275.0391760736142L64,276.37157792014L68,277.4478660334114L72,278.231383078577L76,279.3310277634671L80,280.2601520278229L84,281.03828590819705L88,281.5525360253771L92,282.35797767157163L96,283.02457280388694L100,283.37961588433564L104,284.0662200797949L108,284.7643892061745L112.00000000000001,285.5325280402835L115.99999999999999,285.87933222392866L120,286.47448648068394L124,286.9895102093926L128,287.4136421614427L132,287.7882907097692L136,288.0968088507971L140,288.8952629318002L144,288.9658786744885L148,289.39335548466914L152,289.64674148757257L156,289.88565167672556L160,290.5516983510479L164,290.8298372578144L168,290.77034376614614L172,291.1810715428015L176,291.48607286636525L180,291.69976155999785L184,291.7538213585228L188,292.19905002451577L192,292.3288928771359L196,292.68845630644375L200,292.73166168294983L204,293.0373824962858L208,293.17062550316575L212,293.19190008281475L216,293.36943184637397L220.00000000000003,293.7509047870369L224.00000000000003,293.7823132979518L227.99999999999997,293.93030400548867L231.99999999999997,294.25667218071277L236,294.31138418037705L240,294.4040225604043L244,294.6430158199496L248,294.909997411383L252,294.83397689730356L256,295.0079551147289L260,295.1287764989991L264,295.28880846845124L268,295.4730311945825L272,295.55711670842607L276,295.79912397029557L280,295.82441797631697L284,295.88530805661804L288,296.07770376294536L292,296.1937716839169L296,296.16446426042546L300,296.32795336572684L304,296.3924969340458L308,296.53409179211616L312,296.6938539743782L316,296.7783253877658L320,296.7749296921381L324,296.86394596441005L328,297.0068369960599L332,297.1624239656091L336,297.1538509751207L340,297.2916944916077L344,297.3338485781071L348,297.4552813340686L352,297.54488985380596L356,297.50571834373386L360,297.55852003065L364,297.654060145363L368,297.7574528370619L372,297.81350188694887L376,297.78475367276553L380,297.9389350144084L384,297.885458914999L388,298.040476434124L392,298.0840233834116L396,298.09344180679005" stroke="#4c78a8" stroke-width="2"/></g></g><path class="foreground" aria-hidden="true" d="" display="none"/></g></g></g><path class="foreground" aria-hidden="true" d="" display="none"/></g></g></g></svg>'

render_svg(svg_str)

st.title('Emotion categorization surveys results', 'top')

st.header('Table of contents', 'toc')

st.write("""
[**Forced-choice results**](#title-fc):
- [Participants demographics](#header-fc-dem)
- [Overall results](#header-fc-overall)
- [Results by emotion label](#header-fc-emotions)

[**Free-labeling results**](#title-free):
""")

@st.cache(persist=True, allow_output_mutation=True)
def load_data(path1, path2):
    df = pd.read_csv(path1)
    df_labels = pd.read_csv(path2)
    
    return df, df_labels

@st.cache(persist=True, allow_output_mutation=True)
def count_freq_labels(df, X="all" ):
    if X == "all":
        df_counts = df.stack().reset_index(drop=True).value_counts() # stack as series
        df_counts = df_counts.to_frame('counts') # get value_counts as df
        df_counts['emotion'] = df_counts.index # get index as col
    else:
        df_counts = df[X].reset_index(drop=True).value_counts() # stack as series
        df_counts = df_counts.to_frame('counts') # get value_counts as df
        df_counts[X] = df_counts.index # get index as col

    df_counts = df_counts.reset_index(drop=True) # clean index
    df_counts['percent'] = df_counts['counts'] / df_counts['counts'].sum() # compute percentage
    return df_counts

@st.cache(persist=True, allow_output_mutation=True)
def simple_per_bar(
    df, title='Title', X='percent:Q', Y='emotion:N', \
    width=450, height=250, sort='-x', \
    text_size = 12, label_size = 11, title_size=12, \
    emotion='Some', color1='#0570b0', color2='orange'):
    
    bars = alt.Chart(df, title=title).mark_bar().encode(
        alt.X(X, axis=alt.Axis(format='.0%')),
        y=alt.Y(Y, sort=sort), 
        color=alt.condition(
            alt.datum.emotion == emotion,
            alt.value(color2),
            alt.value(color1)
        ))
    
    text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    fontSize=text_size
    ).encode(
        alt.Text(X, format='.1%')
    )
    
    chart = (bars + text).configure_axis(
            labelFontSize=label_size,
            titleFontSize=title_size).properties(
                width=width, 
                height=height)
    
    
    return chart

@st.cache(persist=True, allow_output_mutation=True)
def simple_count_bar(
    df, title='Title', X='counts:Q', Y='emotion:N', \
    width=450, height=250, sort='-x', \
    text_size = 12, label_size = 11, title_size=12,
    emotion='Some', color1='#0570b0', color2='#orange'):
    
    bars = alt.Chart(df, title=title).mark_bar().encode(
        alt.X(X),
        y=alt.Y(Y, sort=sort), 
        color=alt.condition(
            alt.datum.emotion == emotion,
            alt.value(color2),
            alt.value(color1)
        ))
    
    text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3,  # Nudges text to right so it doesn't appear on top of the bar
    fontSize=text_size
    ).encode(
        alt.Text(X)
    )
    
    chart = (bars + text).configure_axis(
            labelFontSize=label_size,
            titleFontSize=title_size).properties(
                width=width, 
                height=height)
    
    
    return chart

st.title('Forced-choice results', 'title-fc')

st.header('Participants demographics', 'header-fc-dem')

df, df_labels = load_data('clean_data/forced_choice_emotion_uw_students.csv', 'data/emotion_labels.csv')

#####################
### - sex chart - ###

source = count_freq_labels(df, X="sex") 
title = 'Sex | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'sex:N'

chart_sex = simple_per_bar(source, title=title, X=X, Y=Y)

#####################
### - age chart - ###

source = count_freq_labels(df, X="age") 
title = 'Age | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'age:N'

chart_age = simple_per_bar(source, title=title, X=X, Y=Y)

###########################
### - ethnicity chart - ###

source = count_freq_labels(df, X="ethnicity") 
title = 'Ethnicity | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'ethnicity:N'

chart_ethnicity= simple_per_bar(source, title=title, X=X, Y=Y)

###########################
### - education chart - ###

source = count_freq_labels(df, X="formal education") 
title = 'Formal education | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'formal education:N'

chart_formal_education= simple_per_bar(source, title=title, X=X, Y=Y)

##########################
### DEMOGRAPHICS BLOCK ###

col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)

with col1:
    st.subheader("Participants by sex")
    col1.altair_chart(chart_sex, use_container_width=True)
with col2:
    st.subheader("Participants by age group")
    col2.altair_chart(chart_age, use_container_width=True)
with col3:
    st.subheader("Participants by ethnicity")
    col3.altair_chart(chart_ethnicity, use_container_width=True)
with col4:
    st.subheader("Participants by formal education")
    col4.altair_chart(chart_formal_education, use_container_width=True)

st.header('Overall results', 'header-fc-overall')

df_emo_answers = df.loc[:, 'Q2.1':'Q195.1'] # subset photos

##############################
### - overall per chart - ###

source = count_freq_labels(df_emo_answers, X="all") 
title = 'Labels frequency | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_overall_per = simple_per_bar(source, title=title, X=X, Y=Y)


##############################
### - overall cnt chart - ###

source = count_freq_labels(df_emo_answers, X="all") 
title = 'Labels frequency | n = '+ source['counts'].sum().astype(str)
X, Y = 'counts:Q', 'emotion:N'

chart_overall_count = simple_count_bar(source, title=title, X=X, Y=Y)

#############################
### OVERALL RESULTS BLOCK ###

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Selected labels for all images as percentage")
    col1.altair_chart(chart_overall_per, use_container_width=True)

with col2:
    st.subheader("Selected labels for all images as count")
    col2.altair_chart(chart_overall_count, use_container_width=True)

st.header('Results by expected emotion label', 'header-fc-emotions')

@st.cache(persist=True, allow_output_mutation=True)
def emotion_df_formated(df_emo_answers, emotion_label):
    df_emo_cat = df_emo_answers.copy() 
    df_emo_cat_t = df_emo_cat.T # transpose
    df_emo_cat_t['photo_id'] = df_emo_cat_t.index # get index as col
    df_emo_cat_t = df_emo_cat_t.reset_index(drop=True) # clean index
    df_emo_cat_t_labels = pd.concat([df_emo_cat_t, df_labels], axis=1) # add metadata cols
    df_label =  df_emo_cat_t_labels[df_emo_cat_t_labels['label'] == emotion_label]
    return df_label

########################
### - anger chart - ###

emotion = 'anger'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_anger = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

#########################
### - disgust chart - ###

emotion = 'disgust'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_disgust = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

######################
### - fear chart - ###

emotion = 'fear'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_fear = simple_per_bar(source, title=title, X=X, Y=Y,  emotion=emotion.capitalize())

#########################
### - surprise chart- ###

emotion = 'surprise'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_surprise = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

##########################
### - happiness chart- ###

emotion = 'happiness'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_happiness = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())   

#########################
### - sadness chart - ###

emotion = 'sadness'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'


chart_sadness = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

###########################
### - uncertain chart - ###

emotion = 'uncertain'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_uncertain = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

#########################
### - neutral chart - ###

emotion = 'neutral'
df_formated = emotion_df_formated(df_emo_answers, emotion) # subset 'anger' rows
df_formated_ans = df_formated.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
source = count_freq_labels(df_formated_ans, X="all") 
title = 'Expcted label: '+ emotion + ' | n = '+ source['counts'].sum().astype(str)
X, Y = 'percent:Q', 'emotion:N'

chart_neutral = simple_per_bar(source, title=title, X=X, Y=Y, emotion=emotion.capitalize())

########################
### BY EMOTION BLOCK ###

col1, col2 = st.beta_columns(2)
col3, col4 = st.beta_columns(2)
col5, col6 = st.beta_columns(2)
col7, col8 = st.beta_columns(2)

with col1:
    st.subheader("Images depicting 'anger'")
    col1.altair_chart(chart_anger, use_container_width=True)

with col2:
    st.subheader("Images depicting 'disgust'")
    col2.altair_chart(chart_disgust, use_container_width=True)

with col3:
    st.subheader("Images depicting 'fear'")
    col3.altair_chart(chart_fear, use_container_width=True)

with col4:
    st.subheader("Images depicting 'surprise'")
    col4.altair_chart(chart_surprise, use_container_width=True)

with col5:
    st.subheader("Images depicting 'happiness'")
    col5.altair_chart(chart_happiness, use_container_width=True)

with col6:
    st.subheader("Images depicting 'sadness'")
    col6.altair_chart(chart_sadness, use_container_width=True)

with col7:
    st.subheader("Images depicting 'uncertain'")
    col7.altair_chart(chart_uncertain, use_container_width=True)

with col8:
    st.subheader("Images depicting 'neutral'")
    col8.altair_chart(chart_neutral, use_container_width=True)

st.write("""
## Results by expected emotion label, and photo's ethnicity
""")

@st.cache(persist=True, allow_output_mutation=True)
def emotion_df_formated_et(df_emo_answers, emotion_label, ethnicity):
    df_emo_cat = df_emo_answers.copy() 
    df_emo_cat_t = df_emo_cat.T # transpose
    df_emo_cat_t['photo_id'] = df_emo_cat_t.index # get index as col
    df_emo_cat_t = df_emo_cat_t.reset_index(drop=True) # clean index
    df_emo_cat_t_labels = pd.concat([df_emo_cat_t, df_labels], axis=1) # add metadata cols
    df_label =  df_emo_cat_t_labels[(df_emo_cat_t_labels['label'] == emotion_label) & (df_emo_cat_t_labels['ethnicity'] == ethnicity)]
    return df_label

@st.cache(persist=True, allow_output_mutation=True)
def wrapper_chart_emotion(df_emo_answers, emotion, ethnicity):
    df = emotion_df_formated_et(df_emo_answers, emotion,  ethnicity) # subset 'anger' rows
    df_formated_ans = df.drop(['photo_id', 'ethnicity', 'sex','age', 'label', 'url'], axis=1)
    df_count = count_freq_labels(df_formated_ans) # count label freq
    chart = simple_per_bar(
        df_count, \
        title='Expected label: '+ emotion + ' | n = '+ df_count['counts'].sum().astype(str), \
         emotion=emotion.capitalize())
    return chart
    
emotion_option = count_freq_labels(df_emo_answers, X="all")['emotion'].tolist()
emotion_select = st.selectbox("Select emotion", emotion_option)

chart_bipoc = wrapper_chart_emotion(df_emo_answers, emotion_select.lower(), 'bipoc')
chart_white = wrapper_chart_emotion(df_emo_answers, emotion_select.lower(), 'white')

####################################
### BY EMOTION & ETHNICITY BLOCK ###

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader("Images depicting BIPOC")
    col1.altair_chart(chart_bipoc, use_container_width=True)
with col2:
    st.subheader("Images depicting Caucasians")
    col2.altair_chart(chart_white, use_container_width=True)

st.title('Free-labeling results', 'title-free')
