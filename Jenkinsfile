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
for(b = 0; b < builderNodes.size(); b++) {
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
                                sh """
                                git fetch --all
                                git reset --hard origin/genn_4
                                """;
                            }
                        }
                        else {
                            echo "Cloning GeNN";
                            sh "git clone https://github.com/genn-team/genn.git";
                        }

                        // Remove existing virtualenv
                        sh "rm -rf virtualenv";

                        sh "pip install virtualenv";
                        // Create new one
                        echo "Creating virtualenv";
                        sh "virtualenv virtualenv";

                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }
        
                buildStep("Installing Python modules(" + env.NODE_NAME + ")") {
                    // Activate virtualenv and intall packages
                    // **TODO** we shouldn't manually install most of these - they SHOULD get installed when we install pynn_genn
                    sh """
                    . virtualenv/bin/activate
                    pip install nose nose_testconfig coverage codecov "numpy>1.6, < 1.15" scipy
                    """;
                }

                buildStep("Building PyGeNN (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        // Build dynamic LibGeNN
                        // **TODO** only do this stage if anything's changed in GeNN
                        echo "Building LibGeNN";
                        def uniqueLibGeNNBuildMsg = "libgenn_build_" + env.NODE_NAME;

                        // Remove existing logs
                        sh """
                        rm -f ${uniqueLibGeNNBuildMsg}
                        """;
                        
                        // **YUCK** if dev_toolset is in node label - source in newer dev_toolset (CentOS)
                        makeCommand = "";
                        if("dev_toolset" in nodeLabel) {
                            makeCommand += ". /opt/rh/devtoolset-6/enable\n"
                        }
                        
                        // Make LibGeNN and all supported backends with relocatable code, suitable for including in Python module
                        makeCommand += "make RELOCATABLE=1";
                        
                        // Make
                        def makeStatusCode = sh script:makeCommand, returnStatus:true
                        if(makeStatusCode != 0) {
                            setBuildStatus("Building PyGeNN (" + env.NODE_NAME + ")", "FAILURE");
                        }
                        
                        // Archive build message
                        archive uniqueLibGeNNBuildMsg
                        
                        def uniquePluginBuildMsg = "pygenn_plugin_build_" + env.NODE_NAME;

                        // Activate virtualenv, remove existing logs, clean, build module and archive output
                        // **HACK** installing twice as a temporary solution to https://stackoverflow.com/questions/12491328/python-distutils-not-include-the-swig-generated-module
                        echo "Building Python module";
                        script = """
                        . ../virtualenv/bin/activate
                        python setup.py clean --all
                        rm -f ${uniquePluginBuildMsg}
                        python setup.py install 1>> "${uniquePluginBuildMsg}" 2>> "${uniquePluginBuildMsg}"
                        python setup.py install 1>> "${uniquePluginBuildMsg}" 2>> "${uniquePluginBuildMsg}"
                        """
                        def installStatusCode = sh script:script, returnStatus:true
                        if(installStatusCode != 0) {
                            setBuildStatus("Building PyGeNN (" + env.NODE_NAME + ")", "FAILURE");
                        }
                        
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
                        
                        // Remove existing logs
                        sh """
                        rm -f ${uniqueTestOutputMsg}
                        """;
                        
                        // Activate virtualenv, remove log and run tests (keeping return status)
                        def testCommand = """
                        . ../../../virtualenv/bin/activate
                        rm -f .coverage
                        nosetests -s --with-xunit --with-coverage --cover-package=pygenn --cover-package=pynn_genn test_genn.py 1>> "${uniqueTestOutputMsg}" 2>> "${uniqueTestOutputMsg}"
                        """
                        def testStatusCode = sh script:testCommand, returnStatus:true
                        if(testStatusCode != 0) {
                            setBuildStatus("Running tests (" + env.NODE_NAME + ")", "UNSTABLE");
                        }
                        
                        // Activate virtualenv and  convert coverage to XML
                        def coverageCommand = """
                        . ../../../virtualenv/bin/activate
                        coverage xml
                        """
                        def coverageStatusCode = sh script:coverageCommand, returnStatus:true
                        if(coverageStatusCode != 0) {
                            setBuildStatus("Running tests (" + env.NODE_NAME + ")", "UNSTABLE");
                        }
                        
                        archive uniqueTestOutputMsg;
                    }
                    
                    // Switch to PyNN GeNN repository root so codecov uploader works correctly
                    dir("pynn_genn") {
                        // Activate virtualenv and upload coverage
                        sh """
                        . ../virtualenv/bin/activate
                        codecov --token 1460b8f4-e4af-4acd-877e-353c9449111c --file test/system/coverage.xml
                        """
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
