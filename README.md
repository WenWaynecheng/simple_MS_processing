# simple_MS_processing
part of my current master's degree project for non-target analysis at UofA
This is only specifically for LCMS data treatment and only mzML file for now.

This uses part of the pyOpenMS package for detecting the Mass spec peaks 
However, the issue was that the package was more designed for Metabolomics/Proteomics
where the peak alignment park doesnt really do what we usually perform directly on the vendor software
i.e., find a m/z, get EIC, and record the peak area

Although some alignment algorithms were included, a custom alignment method was also included (which might be preferred on my end)


