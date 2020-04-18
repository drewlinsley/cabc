# The Cluttered NIST challenge

This repository contains code to create images of the Cluttered NIST challenge. 

Task design is an extension of the Pathfinder challenge introduced in Linsley et al., 2018. The cluttered NIST challenge differs from the Pathfinder challenge in that the task involves making judgement about whether multiple markers in an image belong to one object instance (a NIST letter), instead of a single connected curve as in the Pathfinder. 

A similar stimulus design called 'cluttered Omniglot' is used in Michaelis et al., 2018. Our challenge differs from cluttered Omniglot because we allow multiple copies of a single category to be present in an image, and we cast the problem as binary categorization instead of pixel-to-pixel segmentation. We believe that this task can be solved sufficiently easily by recognizing a familiar object _and_ by segmenting its instance from the rest of the image, which we believe can be done via a combination of feedforward and feedback mechanisms. The challenge will be very difficult otherwise if only feedforward mechanisms are used.

Code still in the works. Run test.py for a test image.
