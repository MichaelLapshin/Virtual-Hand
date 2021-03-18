using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;

public class UserComponent : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] startingAngles;
    private float[] torquesToApply;

    // Process/training variables
    private bool training;
    private Process process;
    private long timeNS_sinceStart;


    // Start is called before the first frame update
    void Start()
    {
        // Object fetching related logic
        movableLimbs = FindObjectsOfType<HingeJoint>(); // Should return all the finger limbs
        rigidBodies = new Rigidbody[movableLimbs.Length];
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }

        startingAngles = new float[movableLimbs.Length];
        torquesToApply = new float[movableLimbs.Length];

        // Model related logic
        training = true;

        process = new Process();
        process.StartInfo.Arguments = "training_dataset_NAME.txt";

        // Calls python training script.
        if (training == true)
        {
            process.StartInfo.FileName = "ModelTrainer.py";
        }
        else
        {
            // string scriptName = "ModelReadingsMapper.py";
            // string modelName = ""; // Model to load in
            // process.StartInfo.FileName = scriptName + modelName;
        }

        // Starts the process
        process.Start();

        // Interactions with the Python Script
        if (training == true)
        {
            string acknowledgement = process.StandardOutput.ReadLine();
            if (acknowledgement.Equals("Ready") == false)
            {
                Console.Error.Write("Did not receive acknowledgement from Python script.");
                this.enabled = false; // Kills this script.
                return;
            }

            // Obtains starting angles from the python script
            string[] stringBaseAngles = process.StandardOutput.ReadLine().Split(' ');
            for (int i = 0; i < stringBaseAngles.Length; i++)
            {
                startingAngles[i] = float.Parse(stringBaseAngles[i]);
            }

            ResetHandPhysics();
        }
        else
        {
        }
    }

    private void FixedUpdate()
    {
        if (training == true)
        {
            // todo, complete time-syncing logic
            
            // Composes the message to send to the python script
            string toSend = "";
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                if (i != 0)
                {
                    toSend += " ";
                }

                toSend += movableLimbs[i].transform.eulerAngles.x + " " + rigidBodies[i].angularVelocity.x;
            }

            // Sends data to python script
            process.StandardInput.WriteLine(toSend);
            process.StandardInput.Flush();

            string nextCommand = process.StandardOutput.ReadLine();
            if (nextCommand.Equals("Failed"))
            {
                //todo, do the restarting sequence
            }
            else if (nextCommand.Equals("Quit"))
            {
                this.enabled = false; // Kills this script.
                return;
            }
            else if (nextCommand.Equals("Next"))
            {
                // Obtains and applies torques from python script to the limbs
                string[] stringTorques = process.StandardOutput.ReadLine().Split(' ');
                for (int i = 0; i < movableLimbs.Length; i++)
                {
                    rigidBodies[i].AddTorque(new Vector3(float.Parse(stringTorques[i]), 0, 0), ForceMode.Force);
                }
            }
            else
            {
                Console.WriteLine(
                    "Unknown nextCommand sent from python script (" + nextCommand + "). Aborting program.");
                this.enabled = false; // Kills this script.
                return;
            }
        }
        else
        {
        }
    }

    private void ResetHandPhysics()
    {
        for (int i = 0; i < movableLimbs.Length; i++)
        {
            ((GameObject) movableLimbs[i]).transform.eulerAngles.Set(startingAngles[i], 0, 0);
            rigidBodies[i].velocity = Vector3.zero;
            rigidBodies[i].angularVelocity = Vector3.zero;
        }
    }
}