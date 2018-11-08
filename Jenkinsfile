#!groovy​

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda9", "linux", "x86_64", "python27"] as Set,
    ["cuda8", "linux", "x86_64", "python27"] as Set,
    ["cpu_only", "linux", "x86_64", "python27"] as Set,
    ["cuda9", "linux", "x86", "python27"] as Set,
    ["cuda8", "linux", "x86", "python27"] as Set,
    ["cpu_only", "linux", "x86", "python27"] as Set,
    ["cuda9", "mac", "python27"] as Set,
    ["cuda8", "mac", "python27"] as Set,
    ["cpu_only", "mac", "python27"] as Set,
    ["cuda9", "linux", "x86_64", "python3"] as Set,
    ["cuda8", "linux", "x86_64", "python3"] as Set,
    ["cpu_only", "linux", "x86_64", "python3"] as Set,
    ["cuda9", "linux", "x86", "python3"] as Set,
    ["cuda8", "linux", "x86", "python3"] as Set,
    ["cpu_only", "linux", "x86", "python3"] as Set,
    ["cuda9", "mac", "python3"] as Set,
    ["cuda8", "mac", "python3"] as Set,
    ["cpu_only", "mac", "python3"] as Set] 

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------
// Wrapper around setting of GitHUb commit status curtesy of https://groups.google.com/forum/#!topic/jenkinsci-issues/p-UFjxKkXRI
// **NOTE** since that forum post, stage now takes a Closure as the last argument hence slight modification 
void buildStep(String message, Closure closure) {
    stage(message)
    {
        try {
            setBuildStatus(message, "PENDING");
            closure();
        } catch (Exception e) {
            setBuildStatus(message, "FAILURE");
        }
    }
}

void setBuildStatus(String message, String state) {
    // **NOTE** ManuallyEnteredCommitContextSource set to match the value used by bits of Jenkins outside pipeline control
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/genn-team/genn/"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "continuous-integration/jenkins/branch"],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

//--------------------------------------------------------------------------
// Entry point
//--------------------------------------------------------------------------
// Build dictionary of available nodes and their labels
def availableNodes = [:]
for(node in jenkins.model.Jenkins.instance.nodes) {
    if(node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes[node.name] = node.getLabelString().split() as Set
    }
}

// Add master if it has any idle executors
if(jenkins.model.Jenkins.instance.toComputer().countIdle() > 0) {
    availableNodes["master"] = jenkins.model.Jenkins.instance.getLabelString().split() as Set
}

// Loop through the desired builds
def builderNodes = []
for(b in desiredBuilds) {
    // Loop through all available nodes
    for(n in availableNodes) {
        // If, after subtracting this node's labels, all build properties are satisfied
        if((b - n.value).size() == 0) {
            // Add node's name to list of builders and remove it from dictionary of available nodes
            // **YUCK** for some reason tuples aren't serializable so need to add an arraylist
            builderNodes.add([n.key, n.value])
            availableNodes.remove(n.key)
            break
        }
    }
}

//--------------------------------------------------------------------------
// Parallel build step
//--------------------------------------------------------------------------
// **YUCK** need to do a C style loop here - probably due to JENKINS-27421 
def builders = [:]
for(b = 0; b < builderNodes.size; b++) {
    // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
    def nodeName = builderNodes.get(b).get(0)
    def nodeLabel = builderNodes.get(b).get(1)
   
    // Create a map to pass in to the 'parallel' step so we can fire all the builds at once
    builders[nodeName] = {
        node(nodeName) {
            def installationStageName =  "Installation (" + env.NODE_NAME + ")";
 
            // Customise this nodes environment so GeNN environment variable is set and genn binaries are in path
            // **NOTE** these are NOT set directly using env.PATH as this makes the change across ALL nodes which means you get a randomly mangled path depending on node startup order
            withEnv(["GENN_PATH=" + pwd() + "/genn",
                     "PATH+GENN=" + pwd() + "/genn/lib/bin"]) {
                stage(installationStageName) {
                    echo "Checking out GeNN";

                    // Deleting existing checked out version of pynn_genn
                    sh "rm -rf pynn_genn";

                    dir("pynn_genn") {
                        // Checkout pynn_genn here
                        // **NOTE** because we're using multi-branch project URL is substituted here
                        checkout scm
                    }

                    // **NOTE** only try and set build status AFTER checkout
                    try {
                        setBuildStatus(installationStageName, "PENDING");

                        // If GeNN exists
                        if(fileExists("genn")) {
                            echo "Updating GeNN";
                            
                            // Pull from repository
                            dir("genn") {
                                sh "git pull";
                            }
                        }
                        else {
                            echo "Cloning GeNN";
                            sh "git clone -b python_wrapper https://github.com/genn-team/genn.git";
                        }

                        // Remove existing virtualenv
                        sh "rm -rf virtualenv";

                        // Create new one
                        echo "Creating virtualenv";
                        sh "virtualenv --system-site-packages virtualenv";

                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }
        
                buildStep("Installing Python modules(" + env.NODE_NAME + ")") {
                    // Activate virtualenv and intall packages
                    sh """
                    . virtualenv/bin/activate
                    pip install neo pynn quantities nose nose_testconfig lazyarray coverage codecov numpy
                    """;
                }

                buildStep("Building PyGeNN (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        // Build dynamic LibGeNN
                        // **TODO** only do this stage if anything's changed in GeNN
                        echo "Building LibGeNN";
                        def uniqueLibGeNNBuildMsg = "libgenn_build_" + env.NODE_NAME;
                        
                        // **YUCK** if dev_toolset is in node label - source in newer dev_toolset (CentOS)
                        makeCommand = "";
                        libGeNNName = "libgenn_DYNAMIC"
                        if("dev_toolset" in nodeLabel) {
                            makeCommand += ". /opt/rh/devtoolset-6/enable\n"
                        }
                        
                        // Add start of make command
                        makeCommand += "make -f lib/GNUMakefileLibGeNN DYNAMIC=1 ";
                        
                        // Add CPU only options
                        if("cpu_only" in nodeLabel) {
                            makeCommand += "CPU_ONLY=1 ";
                            libGeNNName = "libgenn_CPU_ONLY_DYNAMIC"
                        }
                        
                        // Add remainder of make incantation
                        makeCommand += """
                        LIBGENN_PATH=pygenn/genn_wrapper/ 1> "${uniqueLibGeNNBuildMsg}" 2> "${uniqueLibGeNNBuildMsg}"
                        """
                        
                        // Make
                        sh makeCommand;
                        
                        // If node is a mac, re-label library
                        if("mac" in nodeLabel) {
                            sh "install_name_tool -id \"@loader_path/" + libGeNNName + ".dylib\" pygenn/genn_wrapper/" + libGeNNName + ".dylib";
                        }
                        
                        // Archive build message
                        archive uniqueLibGeNNBuildMsg
                        
                        def uniquePluginBuildMsg = "pygenn_plugin_build_" + env.NODE_NAME;
                        
                        // Activate virtualenv, build module and archive output
                        echo "Building Python module";
                        sh """
                        . ../virtualenv/bin/activate
                        python setup.py install 1>> "${uniquePluginBuildMsg}" 2>> "${uniquePluginBuildMsg}"
                        """
                        
                        archive uniquePluginBuildMsg;
                    }
                }
                
                buildStep("Installing PyNN GeNN (" + env.NODE_NAME + ")") {
                    dir("pynn_genn") {
                        // Activate virtualenv and install PyNN GeNN
                        sh """
                        . ../virtualenv/bin/activate
                        python setup.py install
                        """;
                    }
                }
                
                buildStep("Running tests (" + env.NODE_NAME + ")") {
                    dir("pynn_genn/test/system") {
                        // Generate unique name for message
                        def uniqueTestOutputMsg = "test_output_" + env.NODE_NAME;
                        def uniqueCoverageFile = ".coverage." + env.NODE_NAME;
                        
                        // Activate virtualenv and run tests
                        sh """
                        . ../../../virtualenv/bin/activate
                        nosetests -s --with-xunit --with-coverage --cover-package=pygenn --cover-package=pynn_genn test_genn.py 1>> "${uniqueTestOutputMsg}" 2>> "${uniqueTestOutputMsg}"
                        mv .coverage ${uniqueCoverageFile}
                        """
                        
                        // Archive output
                        archive uniqueTestOutputMsg;
                        
                        // Stash coverage
                        stash name: nodeName + "_coverage", includes: uniqueCoverageFile
                    }
                }

                buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                    // Process JUnit test output
                    junit "pynn_genn/test/**/nosetests.xml";
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders

//--------------------------------------------------------------------------
// Final combination of results
//--------------------------------------------------------------------------
node {
    buildStep("Uploading coverage summary") {
        // Switch to GeNN test directory so git repo (and hence commit etc) can be detected 
        // and so coverage reports gets deleted with rest of GeNN at install-time
        dir("pynn_genn/test/system") {
            // Loop through builders
            def coverageCommand = "coverage combine";
            def anyCoverage = false
            for(b = 0; b < builderNodes.size; b++) {
                // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
                def nodeName = builderNodes.get(b).get(0)
      
                // Unstash coverage
                unstash nodeName + "_coverage"
                
                // If coverage file exists in stash
                if(fileExists(".coverage." + nodeName)) {
                    anyCoverage = true;
                }
                else {
                    echo "Coverage file generated by node:" + nodeName + " not found in stash"
                }
            }
            
            // If any coverage reports were found
            if(anyCoverage) {
                // Activate virtualenv, combine coverage
                sh """
                . ../../virtualenv/bin/activate
                coverage combine
                codecov --token=1460b8f4-e4af-4acd-877e-353c9449111c
                """
            }
            else {
                echo "No coverage reports found"
            }
        }
    }
}