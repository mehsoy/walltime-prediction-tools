#/bin/bash


WORKLOADS="
http://www.cs.huji.ac.il/labs/parallel/workload/l_nasa_ipsc/NASA-iPSC-1993-3.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_lanl_cm5/LANL-CM5-1994-4.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_par/SDSC-Par-1995-3.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_par/SDSC-Par-1996-3.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_ctc_sp2/CTC-SP2-1995-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_ctc_sp2/CTC-SP2-1996-3.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_t3d/LLNL-T3D-1996-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_kth_sp2/KTH-SP2-1996-2.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/SDSC-SP2-1998-4.2-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_lanl_o2k/LANL-O2K-1999-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_osc/OSC-Clust-2000-3.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_blue/SDSC-BLUE-2000-4.2-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sandia_ross/Sandia-Ross-2001-1.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_hpc2n/HPC2N-2002-2.2-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_das2/DAS2-fs0-2003-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_das2/DAS2-fs1-2003-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_das2/DAS2-fs2-2003-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_das2/DAS2-fs3-2003-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_das2/DAS2-fs4-2003-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_ds/SDSC-DS-2004-2.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_lpc/LPC-EGEE-2004-1.2-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_lcg/LCG-2005-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sharcnet/SHARCNET-2005-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_sharcnet/SHARCNET-Whale-2006-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_ubgl/LLNL-uBGL-2006-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_atlas/LLNL-Atlas-2006-2.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_llnl_thunder/LLNL-Thunder-2007-1.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_anl_int/ANL-Intrepid-2009-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_metacentrum/METACENTRUM-2009-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_pik_iplex/PIK-IPLEX-2009-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_ricc/RICC-2010-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_cea_curie/CEA-Curie-2011-2.1-cln.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_intel_netbatch/Intel-NetbatchA-2012-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_intel_netbatch/Intel-NetbatchB-2012-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_intel_netbatch/Intel-NetbatchC-2012-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_intel_netbatch/Intel-NetbatchD-2012-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_unilu_gaia/UniLu-Gaia-2014-2.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_metacentrum2/METACENTRUM-2013-3.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_ciemat_euler/CIEMAT-Euler-2008-1.swf.gz
http://www.cs.huji.ac.il/labs/parallel/workload/l_kit_fh2/KIT-FH2-2016-1.swf.gz
"

WORKLOADDIR=workloads/
if [ -! d "$WORKLOADDIR" ]; then
  # Control will enter here if $DIRECTORY exists.
  mkdir $WORKLOADDIR 
fi

cd $WORKLOADDIR

for item  in $WORKLOADS
do 
	
	wget $item
done

