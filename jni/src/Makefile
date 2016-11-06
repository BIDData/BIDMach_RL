include Makefile.incl

CUDA_OBJS=

CPU_OBJS=BIDMach_ALE.$(OBJ)

.SUFFIXES: .$(OBJ) .c .cpp .cu

all: $(LIBPREPEND)bidmachale$(LIBAPPEND)

cudalib: 

cpulib: $(LIBPREPEND)bidmachale$(LIBAPPEND) 

$(LIBPREPEND)bidmachale$(LIBAPPEND): $(CPU_OBJS)
	$(LD) $(LDFLAGS) $(CPU_OBJS) $(CPU_LIBS) $(OUTFLG)$@

%.$(OBJ) : %.c
	$(CC) $(CFLAGS) $(LAPACK_INCLUDES) $*.c

%.$(OBJ) : %.cpp
	$(GCC) $(CPPFLAGS) $(LAPACK_INCLUDES) $*.cpp

install: installcpulib

installcpulib: $(INSTALLDIR)/$(LIBPREPEND)bidmachale$(LIBAPPEND)

$(INSTALLDIR)/$(LIBPREPEND)bidmachale$(LIBAPPEND): $(LIBPREPEND)bidmachale$(LIBAPPEND)
	mkdir -p $(INSTALL_DIR)
	cp $(LIBPREPEND)bidmachale$(LIBAPPEND) $(INSTALL_DIR)
	cp $(ALE_HOME)/libale$(LIBALEAPPEND)  $(INSTALL_DIR)

jars: cpujar

cpujar: $(BIDMACH_ROOT)/target/BIDMachRL-$(VERSION)-cpu-$(OS)-$(MARCH).jar

$(BIDMACH_ROOT)/target/BIDMachRL-$(VERSION)-cpu-$(OS)-$(MARCH).jar: $(LIBPREPEND)bidmachale$(LIBAPPEND)
	cp $(LIBPREPEND)bidmachale$(LIBAPPEND) $(LIBDIR)
	jar cf $(BIDMACH_ROOT)/target/BIDMachRL-$(VERSION)-cpu-$(OS)-$(MARCH).jar \
	-C $(BIDMACH_ROOT) lib/$(LIBPREPEND)bidmachale$(LIBAPPEND)
	jar uf $(BIDMACH_ROOT)/target/BIDMachRL-$(VERSION)-cpu-$(OS)-$(MARCH).jar \
	-C $(BIDMACH_ROOT) lib/libale$(LIBALEAPPEND)
	rm $(LIBDIR)/$(LIBPREPEND)bidmachale$(LIBAPPEND)

clean:
	rm -f *.$(OBJ) *$(LIBALEAPPEND) *.pdb *.exp *.lib

distclean: clean
	rm -f  *.jnilib Makefile.incl

cleanres:
	rm $(BIDMACH_ROOT)/src/main/resources/lib/*
